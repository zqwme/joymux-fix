# tools/patch_og3_joypad.py
# -*- coding: utf-8 -*-
"""
Edit odroidgo3-joypad node in a DTB:
- dtb -> dts (via dtc), locate node (default /odroidgo3-joypad)
- Update:
  1) amux-channel-mapping  〈property cell order is [RY, RX, LY, LX]〉
     * User supplies the *felt/observed* order for [LX,LY,RX,RY] (e.g. 'LX,LY,RY,RX') or 'unchanged'.
     * We reassign the existing 4 numeric cells back to the canonical axis order [LX,LY,RX,RY],
       then emit them in the property order [RY,RX,LY,LX].
  2) invert-absx/absy/absrx/absry  (toggle semantics)
     * 'unchanged' = keep, else list some of: lx,ly,rx,ry (aliases absx,absy,absrx,absry)
  3) abs_{x,y,rx,ry}-{p,n}-tuning
     * One integer for all, or 8 CSV (x+,x-,y+,y-,rx+,rx-,ry+,ry-)
- dts -> dtb (via dtc). Also writes a unified diff (optional).
"""

import argparse, re, subprocess

# 用户输入与“标准轴顺序”
USER_ORDER = ["LX", "LY", "RX", "RY"]   # 用户始终按这个顺序填写“当前体感”
CANON_ORDER = ["LX", "LY", "RX", "RY"]  # 我们要重建成的标准顺序

# 设备树属性内四个 cell 的真实顺序（重点修正处！）
PROP_ORDER = ["RY", "RX", "LY", "LX"]

# 允许的分隔符
SEP_RE = re.compile(r'[,\s，、;；/／\|]+')


def run(cmd, inp=None):
    r = subprocess.run(cmd, input=inp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{r.stderr.decode()}")
    return r.stdout


def find_node_block(dts_text, node_path):
    """Return (brace_open_idx, brace_close_idx, inner_text) for the node block."""
    needle = node_path.strip("/").split("/")[-1] if node_path.startswith("/") else node_path
    m = None
    for m0 in re.finditer(r'^[ \t]*([A-Za-z0-9_,\-@.]+)\s*\{', dts_text, re.M):
        if needle in m0.group(1):
            m = m0; break
    if not m:
        raise RuntimeError(f"找不到节点名包含 “{needle}” 的块，请检查 node_path")
    brace_open = dts_text.find("{", m.end()-1)
    if brace_open < 0:
        raise RuntimeError("语法错误：未找到左大括号")
    i, depth = brace_open + 1, 1
    while i < len(dts_text):
        c = dts_text[i]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_idx = i; break
        i += 1
    else:
        raise RuntimeError("语法错误：未找到匹配的右大括号")
    inner_start = brace_open + 1
    return brace_open, end_idx, dts_text[inner_start:end_idx]


def upsert_prop(block_text, prop_name, value_line):
    """
    Replace if exists; else insert before first subnode (keep properties before subnodes).
    """
    lines = block_text.splitlines()
    pat_assign = re.compile(r'^\s*' + re.escape(prop_name) + r'\s*=')
    pat_bool   = re.compile(r'^\s*' + re.escape(prop_name) + r'\s*;')
    pat_subnode = re.compile(r'^\s*(?:[A-Za-z0-9_]+:\s*)?[A-Za-z0-9_,\-@.]+\s*\{')

    for i,l in enumerate(lines):
        if pat_assign.match(l) or pat_bool.match(l):
            lines[i] = "    " + value_line.strip()
            return "\n".join(lines)

    insert_at = None
    for i,l in enumerate(lines):
        if pat_subnode.match(l):
            insert_at = i; break

    new_line = "    " + value_line.strip()
    if insert_at is None:
        if lines and lines[-1].strip(): lines.append("")
        lines.append(new_line)
    else:
        lines.insert(insert_at, new_line)
    return "\n".join(lines)


def remove_prop(block_text, prop_name):
    lines = block_text.splitlines()
    pat_assign = re.compile(r'^\s*' + re.escape(prop_name) + r'\s*=')
    pat_bool   = re.compile(r'^\s*' + re.escape(prop_name) + r'\s*;')
    return "\n".join([l for l in lines if not (pat_assign.match(l) or pat_bool.match(l))])


def _split_tokens(s: str):
    return [t for t in SEP_RE.split(s or "") if t]


def parse_user_amux(s):
    """
    Return:
      None -> unchanged
      ["LX","LY","RX","RY"] observed order (must be a permutation)
    """
    raw = (s or "").strip()
    raw_l = raw.lower().replace(" ", "")
    if raw_l in ("", "unchanged", "保持不变", "不修改", "无需改动"):
        return None
    arr = [x.upper() for x in _split_tokens(raw)]
    if set(arr) != set(USER_ORDER) or len(arr) != 4:
        raise ValueError(f"amux_map 必须是 4 项且为 {USER_ORDER} 的一个排列；解析到: {arr}")
    return arr


def parse_invert_tokens(s):
    """
    unchanged -> None；否则返回 要翻转(toggle)的轴 列表（absx/absy/absrx/absry），
    支持 lx/ly/rx/ry 别名与多分隔符。
    """
    raw = (s or "").strip()
    raw_l = raw.lower().replace(" ", "")
    if raw_l in ("", "unchanged", "保持不变", "不修改", "无需改动"):
        return None
    parts = [p.lower() for p in _split_tokens(raw)]
    alias = {
        "lx":"absx", "ly":"absy", "rx":"absrx", "ry":"absry",
        "absx":"absx", "absy":"absy", "absrx":"absrx", "absry":"absry",
    }
    out = []
    for p in parts:
        if p not in alias:
            raise ValueError(f"invert_axes 只允许 'unchanged' 或列出要翻转的轴：lx,ly,rx,ry（兼容 absx...）；解析到: {parts}")
        out.append(alias[p])
    return out


def parse_tuning(s):
    s = (s or "200").replace(" ", "")
    if s.isdigit(): return [s]*8
    parts = s.split(",")
    if len(parts) != 8 or not all(p.strip().isdigit() for p in parts):
        raise ValueError("tuning 需为单个整数或 8 个逗号分隔整数（x+,x-,y+,y-,rx+,rx-,ry+,ry-）")
    return [p.strip() for p in parts]


def dec_to_hex_cell(v):
    v = int(v)
    if v < 0 or v > 0xFFFFFFFF:
        raise ValueError("tuning 超出范围 0..4294967295")
    return f"0x{v:x}"   # 动态宽度十六进制


def get_current_invert_flags(block):
    return {k: bool(re.search(rf'^\s*invert-{k}\s*;', block, re.M))
            for k in ["absx","absy","absrx","absry"]}


def parse_current_amux_cells(block):
    """
    Read 'amux-channel-mapping = <...>;' and return list of 4 ints
    in the property order [RY, RX, LY, LX].
    """
    m = re.search(r'^\s*amux-channel-mapping\s*=\s*<([^>]+)>\s*;', block, re.M | re.I)
    if not m:
        raise RuntimeError("未找到 amux-channel-mapping 属性")
    cells = []
    for tok in re.split(r'[\s,]+', m.group(1).strip()):
        if not tok: continue
        if tok.lower().startswith("0x"):
            cells.append(int(tok, 16))
        else:
            cells.append(int(tok, 10))
    if len(cells) != 4:
        raise ValueError("amux-channel-mapping 解析到的数字数量不是 4")
    return cells  # indices 0..3 correspond to PROP_ORDER [RY,RX,LY,LX]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dtb", required=True, help="Path to input .dtb")
    ap.add_argument("--node_path", default="/odroidgo3-joypad",
                    help="Device tree node path (default: /odroidgo3-joypad)")

    # AMUX：用户按 [LX,LY,RX,RY] 输入“当前感觉顺序”，或 'unchanged'
    ap.add_argument("--amux_csv", required=True,
                    help="Observed order for [LX,LY,RX,RY] (e.g. 'LX,LY,RY,RX') or 'unchanged'")

    # invert：翻转(toggle)语义；'unchanged' 不改
    ap.add_argument("--invert_csv", default="unchanged",
                    help="Toggle list: 'unchanged' or CSV 'lx,rx' (aliases absx,absy,absrx,absry accepted)")

    # tuning：一个数或八个 CSV
    ap.add_argument("--tuning_str", default="200",
                    help="One integer or 8 CSV: x+,x-,y+,y-,rx+,rx-,ry+,ry-")

    args = ap.parse_args()

    # 1) 反编译
    dts = run(["dtc","-I","dtb","-O","dts","-o","-","--", args.in_dtb]).decode("utf-8","replace")
    open("before.dts","w").write(dts)

    # 2) 找节点块
    brace_l, brace_r, block = find_node_block(dts, args.node_path)

    # 3) AMUX：把现有 4 个数字依据“体感顺序”重分配到标准轴顺序，然后按属性顺序写回
    observed = parse_user_amux(args.amux_csv)  # None or 4 axes (USER_ORDER permutation)
    if observed is not None:
        cur_cells = parse_current_amux_cells(block)  # property order [RY,RX,LY,LX]

        # 现状：轴 -> 数值（从属性顺序映射而来）
        old_val_by_axis = { PROP_ORDER[i]: cur_cells[i] for i in range(4) }  # e.g. {'RY':2,'RX':3,'LY':1,'LX':0}

        # 根据用户“体感顺序”，重建标准轴顺序的数值：
        # new_val[axis_in_CANON] = old_val_by_axis[ observed_at_that_position ]
        new_val_by_axis = {}
        for i, axis_pos in enumerate(CANON_ORDER):     # axis_pos traverses LX,LY,RX,RY
            observed_axis = observed[i].upper()        # what user feels at that position
            new_val_by_axis[axis_pos] = old_val_by_axis[observed_axis]

        # 按设备树属性的真实顺序 [RY,RX,LY,LX] 输出
        new_cells = [ new_val_by_axis[a] for a in PROP_ORDER ]
        amux_line = "amux-channel-mapping = <" + " ".join(str(x) for x in new_cells) + ">;"
        block = upsert_prop(block, "amux-channel-mapping", amux_line)

    # 4) invert（翻转语义）
    inv_tokens = parse_invert_tokens(args.invert_csv)
    if inv_tokens is not None:
        cur_flags = get_current_invert_flags(block)
        for t in inv_tokens:  # toggle
            cur_flags[t] = not cur_flags[t]
        def apply_bool(name, enabled, blk):
            return upsert_prop(blk, name, name + ";") if enabled else remove_prop(blk, name)
        block = apply_bool("invert-absx",  cur_flags["absx"],  block)
        block = apply_bool("invert-absy",  cur_flags["absy"],  block)
        block = apply_bool("invert-absrx", cur_flags["absrx"], block)
        block = apply_bool("invert-absry", cur_flags["absry"], block)

    # 5) tuning
    t = parse_tuning(args.tuning_str)
    keys = ["abs_x-p-tuning","abs_x-n-tuning","abs_y-p-tuning","abs_y-n-tuning",
            "abs_rx-p-tuning","abs_rx-n-tuning","abs_ry-p-tuning","abs_ry-n-tuning"]
    for k,v in zip(keys, t):
        block = upsert_prop(block, k, f"{k} = <{dec_to_hex_cell(v)}>;")

    # 6) 组装 & 回编译
    new_dts = dts[:(brace_l+1)] + "\n" + block.strip("\n") + "\n" + dts[brace_r:]
    open("after.dts","w").write(new_dts)
    run(["dtc","-I","dts","-O","dtb","-o","patched.dtb","--","after.dts"])

    # 7) diff（可选，工作流未上传这些文件）
    try:
        diff = run(["diff","-u","before.dts","after.dts"]).decode()
    except Exception as e:
        diff = str(e)
    open("joypad_diff.patch","w").write(diff)


if __name__ == "__main__":
    main()
