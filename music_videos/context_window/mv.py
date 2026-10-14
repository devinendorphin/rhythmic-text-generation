import sys, math, unicodedata, subprocess, warnings, pickle, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import scipy.signal as ss
warnings.filterwarnings('ignore')

FPS = 30
WIDTH, HEIGHT = 1920, 1080
DUR = 149.52
NF = int(DUR * FPS)
CW, CH, BASE = 25, 52, 39          # cell width / height / baseline
COLS = 58
FSIZE = 40
T_END = 147.25                     # music cuts
BREAK0, BREAK1 = 48.35, 49.25      # bass dropout
LOW0, LOW1 = 99.5, 113.0           # low section

# ---------------------------------------------------------------- audio features
F = np.load('feat.npz')
def norm(v, p=95):
    v = np.convolve(v, np.ones(2) / 2, 'same')
    return np.clip(v / (np.percentile(v, p) + 1e-9), 0, 1.5)
bass = norm(F['bass']); hi = norm(F['hi']); flux = norm(F['flux']); rms = norm(F['rms'])
def pad(v):
    return np.r_[v, np.zeros(NF + 2 - len(v))] if len(v) < NF + 2 else v
bass, hi, flux, rms = map(pad, (bass, hi, flux, rms))
hi_s = np.convolve(hi, np.ones(9) / 9, 'same')
pk, props = ss.find_peaks(F['bass'], distance=9, height=np.median(F['bass']) * 2.2)
kick_str = np.clip(props['peak_heights'] / np.percentile(props['peak_heights'], 90), 0, 1.2)
kicks = [(int(p), float(s)) for p, s in zip(pk, kick_str)
         if not (BREAK0 * FPS <= p <= BREAK1 * FPS) and p < T_END * FPS]
kick_pulse = np.zeros(NF + 2)
for p, s in kicks:
    n = np.arange(0, 20)
    idx = p + n; m = idx < len(kick_pulse)
    kick_pulse[idx[m]] = np.maximum(kick_pulse[idx[m]], s * np.exp(-n[m] / 4.0))

RND = np.random.default_rng(7).random(1 << 20)
def h(*a):
    k = 0
    for i, x in enumerate(a):
        k = k * 1000003 + int(x) * (7919 + 104729 * i)
    return RND[k & ((1 << 20) - 1)]

def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

# ---------------------------------------------------------------- text → clusters
def clusters(s):
    out = []
    for c in s:
        cat = unicodedata.category(c)
        if out and (cat in ('Mn', 'Me', 'Cf') or 0xFE00 <= ord(c) <= 0xFE0F):
            out[-1] += c
        else:
            out.append(c)
    return out

raw = open('lyrics.txt', encoding='utf-8').read().rstrip('\n')
paras = raw.split('\n')

# ---------------------------------------------------------------- fonts / atlas
CHAIN = [('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 0),
         ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 0),
         ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 0),
         ('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 0),
         ('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 0),
         ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 0),
         ('/usr/share/fonts/opentype/unifont/unifont.otf', 0),
         ('/usr/share/fonts/opentype/unifont/unifont_upper.otf', 0),
         ('/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf', 0)]
_cmaps = None
def cmaps():
    global _cmaps
    if _cmaps is None:
        _cmaps = [set(TTFont(p, fontNumber=i, lazy=True).getBestCmap().keys()) for p, i in CHAIN]
    return _cmaps

_fonts = {}
def font(k, size):
    key = (k, size)
    if key not in _fonts:
        p, i = CHAIN[k]
        _fonts[key] = ImageFont.truetype(p, 109 if 'Emoji' in p else size, index=i)
    return _fonts[key]

def visible_cps(cl):
    return [c for c in cl if unicodedata.category(c) != 'Cf' and not 0xFE00 <= ord(c) <= 0xFE0F]

def render_cluster(cl, W=CW, H=CH, base=BASE, size=FSIZE):
    cps = visible_cps(cl)
    if not cps or cl == ' ':
        return np.zeros((H, W), np.float32)
    cm = cmaps()
    k = next((j for j, m in enumerate(cm) if all(ord(c) in m for c in cps)), None)
    if k is None:
        k = next((j for j, m in enumerate(cm) if ord(cps[0]) in m), None)
        if k is not None:
            cps = [c for c in cps if ord(c) in cm[k]]
    if k is None:  # tofu
        t = np.zeros((H, W), np.float32)
        x0, x1, y0, y1 = int(W * .18), int(W * .82), int(base - size * .68), base
        t[y0:y1, x0:x0 + 2] = t[y0:y1, x1 - 2:x1] = 1
        t[y0:y0 + 2, x0:x1] = t[y1 - 2:y1, x0:x1] = 1
        return t
    s = ''.join(cps)
    if 'Emoji' in CHAIN[k][0]:
        im = Image.new('RGBA', (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), s, font=font(k, 109), embedded_color=True)
        a = np.asarray(im).astype(np.float32) / 255
        lum = (a[..., :3].mean(2) * 0.6 + 0.4) * a[..., 3]
        ys, xs = np.nonzero(a[..., 3] > 0.05)
        if len(xs) == 0:
            return np.zeros((H, W), np.float32)
        crop = lum[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sc = min((W - 2) / crop.shape[1], (size * .8) / crop.shape[0])
        r = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        t = np.zeros((H, W), np.float32)
        y = base - r.shape[0] + 3; x = (W - r.shape[1]) // 2
        t[y:y + r.shape[0], x:x + r.shape[1]] = r
        return t
    cvw = W * 5
    im = Image.new('L', (cvw, H), 0)
    ImageDraw.Draw(im).text((2 * W, base), s, font=font(k, size), fill=255, anchor='ls')
    a = np.asarray(im).astype(np.float32) / 255
    cols = np.nonzero(a.max(0) > 0.02)[0]
    if len(cols) == 0:
        return np.zeros((H, W), np.float32)
    x0, x1 = cols.min(), cols.max() + 1
    if k == 0 and x0 >= 2 * W - 3 and x1 <= 3 * W + 3:
        return a[:, 2 * W:3 * W].copy()
    iw = x1 - x0
    if iw <= W - 2:
        cx = (x0 + x1) // 2
        return a[:, cx - W // 2: cx - W // 2 + W].copy()
    sc = (W - 2) / iw
    r = cv2.resize(a[:, x0:x1], (W - 2, max(1, int(H * sc))), interpolation=cv2.INTER_AREA)
    t = np.zeros((H, W), np.float32)
    yo = int(base - base * sc)
    t[yo:yo + r.shape[0], 1:W - 1] = r[:H - yo]
    return t

# ---------------------------------------------------------------- layout
# blocks: (list of paragraph indices, t0, t1, style)
def para_idx(prefix):
    return next(i for i, p in enumerate(paras) if p.startswith(prefix))
SCHED = [
    (para_idx('As this'), 4.5, 24.4, 'flow'),
    (para_idx('[Haitians'), 25.0, 25.0, 'header'),
    (para_idx('Back when'), 25.7, 47.6, 'flow'),
    (para_idx('/'), BREAK0, BREAK0, 'slash'),
    (para_idx('When m'), 49.4, 81.2, 'flow'),
    (para_idx('Pocuman'), 82.4, 91.6, 'plain'),
    (para_idx('Dr. Pit'), 92.4, 99.2, 'plain'),
    (para_idx('RRCF'), 113.0, 127.6, 'flow'),
    (para_idx('Hije'), 128.0, 136.8, 'flow'),
    (para_idx('>>'), 137.2, 145.4, 'flow'),
]

def build():
    glyphs = []   # dict: cl, line, col, t (frame float), kind, ascii
    line = 0
    prev_para = None
    for pi, t0, t1, style in SCHED:
        if prev_para is not None:
            gap = pi - prev_para  # blank lines between paragraphs in source
            line += 1 + (1 if gap > 1 else 0)
        prev_para = pi
        cls = clusters(paras[pi])
        # word wrap
        words, cur = [], []
        for c in cls:
            cur.append(c)
            if c == ' ':
                words.append(cur); cur = []
        if cur: words.append(cur)
        col = 0; placed = []
        for wd in words:
            wl = len(wd) - (1 if wd[-1] == ' ' else 0)
            if col + wl > COLS and col > 0 and wl <= COLS:
                line += 1; col = 0
            for c in wd:
                if col >= COLS:
                    line += 1; col = 0
                if c == ' ' and col == 0:
                    continue
                placed.append([c, line, col]); col += 1
        n = len(placed)
        f0, f1 = t0 * FPS, t1 * FPS
        if style in ('header', 'slash'):
            times = [f0 + (0.0 if style == 'slash' else j * 0.12) for j in range(n)]
        else:
            fr = np.arange(int(f0), int(f1) + 1)
            if style == 'plain':
                w = 0.55 + 0.45 * flux[fr] + 0.0
            else:
                w = 0.08 + flux[fr] ** 2.2 + 0.25 * bass[fr] ** 2
            W = np.cumsum(w); W = W / W[-1]
            times = [float(fr[min(len(fr) - 1, np.searchsorted(W, (j + 1) / n))]) for j in range(n)]
            # keep strictly increasing-ish within a frame: spread sub-frame
            times = [t + (j % 3) * 0.01 for j, t in enumerate(times)]
        for (c, ln, cl_), tt in zip(placed, times):
            glyphs.append(dict(cl=c, line=ln, col=cl_, t=tt, kind=style,
                               ascii=all(ord(x) < 128 for x in c)))
        if style in ('header', 'slash', 'plain'):
            pass
    return glyphs

GL = build()
NG = len(GL)
G_t = np.array([g['t'] for g in GL]); G_line = np.array([g['line'] for g in GL]); G_col = np.array([g['col'] for g in GL])
G_space = np.array([g['cl'] == ' ' for g in GL]); G_ascii = np.array([g['ascii'] for g in GL])
UNIQ = sorted(set(g['cl'] for g in GL))
U_idx = {c: i for i, c in enumerate(UNIQ)}
G_u = np.array([U_idx[g['cl']] for g in GL])
NOISE_POOL = np.array([U_idx[c] for c in UNIQ if not all(ord(x) < 128 for x in c)])
TEXT_POOL = np.array([U_idx[c] for c in UNIQ if c != ' '])

# ---------------------------------------------------------------- per-frame timeline (sequential)
def rows_at(t):
    r = 12.0
    r -= 6.0 * sstep(LOW0, LOW0 + 6.0, t)            # shrink to 6
    r += 1.0 * sstep(LOW1 - 0.3, LOW1 + 0.4, t)      # snap open to 7
    return r

def timeline():
    top = np.zeros(NF + 2); forget = 0.0; topv = 0.0
    cur_line = np.zeros(NF + 2, int); cur_col = np.zeros(NF + 2, int); last_t = np.full(NF + 2, -1e9)
    order = np.argsort(G_t)
    j = 0; cl, cc = 0, 0
    for f in range(NF + 2):
        while j < NG and G_t[order[j]] <= f:
            g = order[j]; cl, cc = G_line[g], G_col[g] + 1; last_t[f] = G_t[g]; j += 1
        if last_t[f] < 0 and f > 0: last_t[f] = last_t[f - 1]
        if cc >= COLS: cl, cc = cl + 1, 0
        cur_line[f], cur_col[f] = cl, cc
        R = int(math.floor(rows_at(f / FPS) + 1e-6))
        target = max(forget, cl - R + 1)
        forget = target
        topv += (target - topv) * 0.16
        top[f] = topv
    return top, cur_line, cur_col, last_t

if os.path.exists('tl.pkl') and os.path.getmtime('tl.pkl') > os.path.getmtime(__file__) and os.path.getmtime('tl.pkl') > os.path.getmtime('lyrics.txt'):
    TOP, CUR_L, CUR_C, LAST_T, LEAVE = pickle.load(open('tl.pkl', 'rb'))
else:
    TOP, CUR_L, CUR_C, LAST_T = timeline()
    # frame each line leaves the window
    maxline = G_line.max() + 1
    LEAVE = np.full(maxline + 1, 10 ** 9)
    for L in range(maxline + 1):
        w = np.nonzero(TOP >= L + 0.55)[0]
        if len(w): LEAVE[L] = w[0]
    pickle.dump((TOP, CUR_L, CUR_C, LAST_T, LEAVE), open('tl.pkl', 'wb'))

# attention events: (frame, [glyph targets])
def att_count(t):
    if t < 4.5: return 0
    if t < 25: return 1
    if t < 48: return 2
    if t < 82: return 3
    if t < LOW1: return 4
    return 4
ATT = []
for p, s in kicks:
    n = att_count(p / FPS)
    if n == 0: continue
    vis = np.nonzero((G_t < p - 2) & (~G_space) & (G_line >= np.ceil(TOP[p]) ) & (G_line < LEAVE.size) )[0]
    vis = vis[LEAVE[G_line[vis]] > p + 10]
    if len(vis) == 0: continue
    rng = np.random.default_rng(p)
    tg = rng.choice(vis, size=min(n, len(vis)), replace=False)
    ATT.append((p, s, tg))

# ---------------------------------------------------------------- rendering
def lerp(a, b, t): return a + (b - a) * t
HOT = np.array([1.0, 0.92, 0.74], np.float32)
WARM = np.array([0.88, 0.86, 0.80], np.float32)
COLD = np.array([0.30, 0.58, 0.64], np.float32)
AMBER = np.array([1.0, 0.62, 0.22], np.float32)
BG = np.array([0.020, 0.024, 0.032], np.float32)

class R:
    def __init__(self):
        self.atlas = np.stack([render_cluster(c) for c in UNIQ])
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        d = np.sqrt(((xx - WIDTH / 2) / (WIDTH * .62)) ** 2 + ((yy - HEIGHT / 2) / (HEIGHT * .62)) ** 2)
        self.vig3 = np.repeat(np.clip(1 - d * d * .55, 0.25, 1)[..., None], 3, 2).astype(np.float32)
        g = np.random.default_rng(3)
        self.grain3 = [np.repeat(cv2.resize(g.normal(0, 0.012, (HEIGHT // 2, WIDTH // 2)).astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)[..., None], 3, 2) for _ in range(8)]
        x = np.arange(4096) / 1024.0
        self.lut = np.clip(np.maximum(x, 0) / (1 + np.maximum(x, 0) * 0.35) * 1.18 * 255, 0, 255).astype(np.uint8)
        self.big = {}
        self.small = np.stack([cv2.resize(a, (CW // 2 + 1, CH // 2), interpolation=cv2.INTER_AREA) for a in self.atlas])

    def bigglyph(self, u, size=760):
        if u not in self.big:
            self.big[u] = render_cluster(UNIQ[u], W=int(size * .9), H=int(size * 1.25), base=int(size * .95), size=size)
        return self.big[u]

    def frame(self, f):
        t = f / FPS
        img = np.empty((HEIGHT, WIDTH, 3), np.float32); img[:] = BG
        kp = float(kick_pulse[f])
        endfade = 1 - sstep(149.0, 149.5, t)
        brk = sstep(BREAK0 - 0.05, BREAK0 + 0.05, t) * (1 - sstep(BREAK1, BREAK1 + 0.6, t))
        out = sstep(T_END, T_END + 1.6, t)

        # window geometry
        Rw = rows_at(t)
        tw = COLS * CW; th = Rw * CH
        padx, pady = 36, 30
        wx0 = (WIDTH - tw) / 2; wy0 = (HEIGHT - th) / 2 - 10
        # intro: border draws itself
        intro = sstep(0.4, 4.2, t)

        # ---- giant background glyph (per 2-beat bar)
        bar = int((t - 0.25) // 1.0)
        if t < BREAK1 + 0.3 and t >= BREAK0:
            u = U_idx['/']; inten = 0.20 * brk + 0.02
        else:
            typed = np.nonzero((G_t < f) & (~G_space))[0]
            if len(typed) and t < T_END + 0.5:
                pick = typed[int(h(bar, 11) * min(len(typed), 60)) - min(len(typed), 60)] if len(typed) else None
                u = G_u[pick]
                ph = ((t - 0.25) % 1.0)
                inten = (0.022 + 0.03 * kp) * (1 - out) * (0.35 if t < 25 else 1)
            else:
                u = None; inten = 0
        if u is not None and inten > 0.002:
            bg = self.bigglyph(u)
            gh, gw = bg.shape
            sc = 1.0 + 0.04 * kp + 0.02 * ((t % 1.0))
            if abs(sc - 1) > 0.005:
                bg = cv2.resize(bg, (int(gw * sc), int(gh * sc)))
                gh, gw = bg.shape
            x = int(WIDTH / 2 - gw / 2 + (h(bar, 3) - .5) * 500); y = int(HEIGHT / 2 - gh / 2 + 40)
            xs0, ys0 = max(0, x), max(0, y); xs1, ys1 = min(WIDTH, x + gw), min(HEIGHT, y + gh)
            if xs1 > xs0 and ys1 > ys0:
                col = AMBER if brk > 0.1 else COLD
                img[ys0:ys1, xs0:xs1] += bg[ys0 - y:ys1 - y, xs0 - x:xs1 - x, None] * col * inten

        # ---- particles from forgotten lines
        top = TOP[f]
        gone = np.nonzero((LEAVE[G_line] <= f) & (LEAVE[G_line] > f - 5 * FPS) & (~G_space) & (G_t < LEAVE[G_line]))[0]
        for g in gone:
            age = (f - LEAVE[G_line[g]]) / FPS
            a = math.exp(-age / 1.4) * 0.55 * endfade
            if a < 0.02: continue
            vx = (h(g, 1) - .5) * 70; vy = -25 - h(g, 2) * 70
            x = wx0 + G_col[g] * CW + vx * age + math.sin(age * 2 + g) * 8 * age
            y = wy0 - CH * 0.6 + vy * age - 9 * age * age
            tile = self.small[G_u[g]]
            xi, yi = int(x), int(y)
            th_, tw_ = tile.shape
            if 0 <= xi < WIDTH - tw_ and 0 <= yi < HEIGHT - th_:
                img[yi:yi + th_, xi:xi + tw_] += tile[..., None] * COLD * a

        # ---- text
        ox = wx0; oy = wy0 - (top % 1.0) * CH  # smooth scroll
        base_line = int(math.floor(top))
        vis = np.nonzero((G_t <= f) & (G_line >= base_line) & (G_line <= base_line + int(Rw) + 1))[0]
        # attention boost
        boost = {}
        for p, s, tg in ATT:
            if p <= f < p + 30:
                a = s * math.exp(-(f - p) / 7.0)
                for g in tg: boost[g] = max(boost.get(g, 0), a)
        p_fl = (0.004 + 0.10 * hi_s[f] ** 2) * (0.4 if t < 49 else 1.0) * (0.3 if 82 < t < LOW1 else 1)
        if t > LOW1: p_fl *= 1.8
        nvis = len(vis)
        endrank = np.argsort(np.argsort(G_t[vis])) if nvis else []
        for k, g in enumerate(vis):
            if G_space[g] and G_t[g] > f - 2: pass
            row = G_line[g] - top
            y = int(round(wy0 + row * CH)); x = int(round(ox + G_col[g] * CW))
            if row < -0.98 or row > Rw - 0.02: continue
            age = (f - G_t[g]) / FPS
            u = G_u[g]
            # sampling scramble when born
            if age < 0.14 and G_kind_scr[g]:
                u = TEXT_POOL[int(h(g, f, 5) * len(TEXT_POOL))]
            elif not G_ascii[g] and h(g, f // 2, 9) < p_fl:
                u = NOISE_POOL[int(h(g, f, 6) * len(NOISE_POOL))]
            # colour
            hot = math.exp(-age / 0.22)
            body = lerp(WARM, COLD, 1 - math.exp(-age / 9.0))
            inten = 0.40 + 0.60 * math.exp(-age / 14.0)
            c = lerp(body * inten, HOT * 1.7, hot)
            if G_kind[g] == 'plain':
                c = lerp(WARM * 0.95, HOT * 1.6, hot)
            b = boost.get(g, 0)
            if b: c = c + AMBER * b * 1.3
            # fade at top edge / bottom edge
            edge = min(1.0, (row + 1.0) / 1.0) if row < 0 else min(1.0, (Rw - row) / 0.6)
            m = edge * (1 - 0.88 * brk * (G_kind[g] != 'slash'))
            if G_kind[g] == 'slash': m *= 1 + 1.5 * brk
            # ending: decay oldest → newest
            if t > T_END:
                d = T_END + 0.15 + 1.3 * endrank[k] / max(1, nvis - 1)
                if t > d:
                    m *= max(0, 1 - (t - d) / 0.25)
                    if h(g, f, 8) < 0.6: u = NOISE_POOL[int(h(g, f, 4) * len(NOISE_POOL))]
            m *= endfade
            if m <= 0.01: continue
            tile = self.atlas[u]
            if G_kind[g] == 'header':
                hb = math.exp(-age / 0.5)
                bgc = AMBER * (0.55 + 0.6 * hb) * m
                y0c, y1c = max(0, y + 6), min(HEIGHT, y + CH - 4)
                img[y0c:y1c, x:x + CW] += bgc
                c = -AMBER * 0.0 + BG
                img[y0c:y1c, x:x + CW] -= tile[y0c - y:y1c - y, :, None] * (bgc - BG * 0)
                continue
            y0c, y1c = max(0, y), min(HEIGHT, y + CH)
            if y1c > y0c:
                img[y0c:y1c, x:x + CW] += tile[y0c - y:y1c - y, :, None] * (c * m)

        # ---- attention arcs
        cxp = ox + CUR_C[f] * CW + CW / 2; cyp = wy0 + (CUR_L[f] - top) * CH + CH * .45
        layer = None
        for p, s, tg in ATT:
            if not (p <= f < p + 26): continue
            a = s * math.exp(-(f - p) / 6.0) * endfade
            if layer is None: layer = np.zeros((HEIGHT, WIDTH), np.uint8)
            for g in tg:
                row = G_line[g] - top
                if row < -0.5: continue
                tx = ox + G_col[g] * CW + CW / 2; ty = wy0 + row * CH + CH * .45
                dx, dy = tx - cxp, ty - cyp; dist = math.hypot(dx, dy) + 1
                mx, my = (cxp + tx) / 2, (cyp + ty) / 2 - 0.35 * dist - 20
                # grow arc from cursor
                grow = min(1.0, (f - p + 1) / 4.0)
                ts = np.linspace(0, grow, 48)
                px = (1 - ts) ** 2 * cxp + 2 * (1 - ts) * ts * mx + ts ** 2 * tx
                py = (1 - ts) ** 2 * cyp + 2 * (1 - ts) * ts * my + ts ** 2 * ty
                pts = np.stack([px, py], 1).astype(np.int32)[None]
                cv2.polylines(layer, pts, False, int(min(255, 200 * a)), 1, cv2.LINE_AA)
        if layer is not None:
            img += (layer.astype(np.float32) / 255)[..., None] * AMBER * 0.9

        # ---- cursor
        since = (f - LAST_T[f]) / FPS
        if t < T_END + 0.1:
            on = since < 0.35 or ((t - 0.25) % 1.0) < 0.5
        else:
            on = (t > T_END + 1.5) and (((t - T_END) % 0.5) < 0.25 or t > 148.6)
        if t < 0.9: on = False
        if on and endfade > 0:
            row = CUR_L[f] - top
            if -0.2 < row < Rw:
                x = int(round(ox + CUR_C[f] * CW)); y = int(round(wy0 + row * CH))
                img[y + 9:y + CH - 7, x + 2:x + CW - 1] += HOT * (0.9 + 0.5 * kp) * endfade

        # ---- window frame
        fr_i = (0.16 + 0.42 * kp * (1 - brk)) * intro * endfade * (1 - 0.6 * out) + 0.25 * brk
        x0, y0 = int(wx0 - padx), int(wy0 - pady); x1, y1 = int(wx0 + tw + padx), int(wy0 + th + pady)
        fl = np.zeros((HEIGHT, WIDTH), np.uint8)
        L = int(60 + 0 * kp)
        # corner brackets
        for (cx, cy, sx, sy) in [(x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)]:
            ln = int(L * intro)
            if ln > 0:
                cv2.line(fl, (cx, cy), (cx + sx * ln, cy), 255, 2, cv2.LINE_AA)
                cv2.line(fl, (cx, cy), (cx, cy + sy * ln), 255, 2, cv2.LINE_AA)
        # thin full border (draws in from corners)
        e = intro
        mx_, my_ = (x0 + x1) // 2, (y0 + y1) // 2
        for (a_, b_) in [((x0, y0), (x1, y0)), ((x0, y1), (x1, y1))]:
            hw = int((x1 - x0) / 2 * e)
            cv2.line(fl, (x0, a_[1]), (x0 + hw, a_[1]), 90, 1, cv2.LINE_AA)
            cv2.line(fl, (x1 - hw, a_[1]), (x1, a_[1]), 90, 1, cv2.LINE_AA)
        for xx in (x0, x1):
            hh = int((y1 - y0) / 2 * e)
            cv2.line(fl, (xx, y0), (xx, y0 + hh), 90, 1, cv2.LINE_AA)
            cv2.line(fl, (xx, y1 - hh), (xx, y1), 90, 1, cv2.LINE_AA)
        img += (fl.astype(np.float32) / 255)[..., None] * np.array([0.62, 0.74, 0.80], np.float32) * fr_i

        # ---- status line: tokens in window
        ntok = int(((G_t <= f) & (G_line >= math.ceil(top - 0.3)) & ~G_space).sum())
        if t > T_END: ntok = int(ntok * max(0, 1 - (t - T_END) / 1.5))
        stat = f'{ntok:>5} / 16384'
        sy_ = y1 + 14
        sx_ = x1 - len(stat) * 13
        si = 0.20 * intro * endfade
        for ch in stat:
            if ch != ' ':
                tile = self.small[U_idx[ch]] if ch in U_idx else None
                if tile is None:
                    if ch not in self.big: self.big[ch] = cv2.resize(render_cluster(ch), (CW // 2 + 1, CH // 2), interpolation=cv2.INTER_AREA)
                    tile = self.big[ch]
                img[sy_:sy_ + tile.shape[0], sx_:sx_ + tile.shape[1]] += tile[..., None] * np.array([0.6, 0.75, 0.8], np.float32) * si
            sx_ += 13

        # ---- glow
        small = cv2.resize(img, (WIDTH // 4, HEIGHT // 4), interpolation=cv2.INTER_AREA)
        g1 = cv2.GaussianBlur(small, (0, 0), 3)
        g2 = cv2.GaussianBlur(small, (0, 0), 14)
        glow = cv2.resize(g1 * (0.9 + 0.8 * kp) + g2 * (1.1 + 1.2 * kp), (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
        cv2.scaleAdd(glow, 0.75, img, dst=img)
        cv2.multiply(img, self.vig3, dst=img)
        # ---- row tears + chromatic aberration (stronger late)
        late = 0.35 + 0.65 * sstep(LOW1 - 0.5, LOW1 + 0.5, t)
        if t < T_END and t > 25:
            ntear = int(hi[f] ** 3 * 4 * late + (1 if h(f, 21) < 0.08 * late else 0))
            for i in range(ntear):
                yy = int(h(f, i, 22) * HEIGHT); hh = int(4 + h(f, i, 23) * 40)
                dx = int((h(f, i, 24) - .5) * 60 * late)
                img[yy:yy + hh] = np.roll(img[yy:yy + hh], dx, axis=1)
        ca = int(round(kp * 7 * late * (1 - out)))
        if ca:
            img[..., 0] = np.roll(img[..., 0], ca, axis=1)
            img[..., 2] = np.roll(img[..., 2], -ca, axis=1)

        # ---- grain + tone
        cv2.add(img, self.grain3[f % 8], dst=img)
        if t > 149.0: img *= endfade
        np.clip(img, 0, 3.999, out=img)
        return self.lut[(img * 1024).astype(np.uint16)]

G_kind = [g['kind'] for g in GL]
G_kind_scr = np.array([k != 'slash' for k in G_kind])

if __name__ == '__main__':
    mode = sys.argv[1]
    r = R()
    if mode == 'still':
        for ts in sys.argv[2:]:
            f = int(float(ts) * FPS)
            Image.fromarray(r.frame(f)).save(f'still_{ts}.png')
    elif mode == 'info':
        print('glyphs', NG, 'lines', G_line.max() + 1, 'uniq', len(UNIQ), 'kicks', len(kicks), 'att', len(ATT))
        Image.fromarray((np.concatenate([np.concatenate(list(r.atlas[i:i + 40]), 1) for i in range(0, len(UNIQ) - 39, 40)], 0) * 255).astype(np.uint8)).save('atlas.png')
    elif mode == 'range':
        a, b, outp = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                              '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                              '-crf', '17', '-pix_fmt', 'yuv420p', '-tune', 'grain', outp], stdin=subprocess.PIPE)
        for f in range(a, b):
            p.stdin.write(r.frame(f).tobytes())
            if f % 150 == 0: print(outp, f, flush=True)
        p.stdin.close(); p.wait()
