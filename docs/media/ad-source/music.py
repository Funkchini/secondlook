"""SecondLook ad music v2: upbeat 120 BPM tech-pop launch bed with whooshes, risers and impacts.
Everything is synthesised here (no samples). Timeline matches ad.html (30 s)."""
import numpy as np, wave
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000; D = 30.0; N = int(SR * D)
rng = np.random.default_rng(11)
BUS = {k: np.zeros((N, 2)) for k in ("drums", "music", "sfx")}
BEAT = 0.5                       # 120 BPM
DROP = 7.0                       # kick enters with the teal sweep / logo
KICK_END = 26.0                  # kick out before the big check impact
mid = lambda m: 440 * 2 ** ((m - 69) / 12)


def add(bus, sig, start, pan=0.0, g=1.0):
    i = int(round(start * SR)); n = min(len(sig), N - i)
    if n <= 0 or i < 0: return
    sig = sig[:n] * g
    BUS[bus][i:i + n, 0] += sig * np.sqrt(0.5 * (1 - pan))
    BUS[bus][i:i + n, 1] += sig * np.sqrt(0.5 * (1 + pan))


def tt(d): return np.arange(int(d * SR)) / SR
def filt(x, kind, f, order=2):
    return sosfilt(butter(order, f, btype=kind, fs=SR, output="sos"), x)
def saw(f, d, harm=24):
    t = tt(d); s = np.zeros_like(t)
    for k in range(1, harm + 1):
        if f * k > SR / 2.2: break
        s += np.sin(2 * np.pi * f * k * t) / k
    return s * 0.6
def adsr(n, a, d, s, r, hold):
    t = np.arange(n) / SR
    e = np.where(t < a, t / a, np.where(t < a + d, 1 - (1 - s) * (t - a) / d, s))
    rel = np.clip((t - hold) / r, 0, 1); return e * (1 - rel)
def noise(d): return rng.standard_normal(int(d * SR))


# ---------------- instruments ----------------
def kick():
    t = tt(0.42); f = 45 + 110 * np.exp(-t * 32)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7.5)
    s[:240] += filt(noise(0.005), "high", 3000)[:240] * 0.5 if len(s) > 240 else 0
    return np.tanh(s * 1.6)
def clap():
    s = np.zeros(int(0.3 * SR))
    for k, o in enumerate([0, 0.011, 0.022]):
        n = noise(0.25) * np.exp(-tt(0.25) * (45 if k < 2 else 16)); i = int(o * SR)
        s[i:i + len(n)] += n[:len(s) - i]
    return filt(s, "band", [900, 4000]) * 0.9
def hat(open_=False):
    d = 0.22 if open_ else 0.05
    return filt(noise(d), "high", 7000) * np.exp(-tt(d) * (14 if open_ else 70))
def pluck(f, d=0.28):
    s = saw(f, d, 16); return filt(s, "low", 4200) * np.exp(-tt(d) * 13)
def bass(f, d=0.22):
    s = saw(f, d, 12) + 0.5 * np.sin(2 * np.pi * f / 2 * tt(d))
    return filt(s, "low", 900) * adsr(len(s), 0.004, 0.08, 0.6, 0.05, d - 0.05)
def pad(notes, d):
    s = np.zeros(int(d * SR))
    for m in notes:
        for det in (-0.12, 0, 0.12):
            s += saw(mid(m + det), d, 10)
    return filt(s / (3 * len(notes)), "low", 2600) * adsr(len(s), 0.25, 0.3, 0.8, 0.4, d - 0.4)
def whoosh(d=0.6, f0=300, f1=6000, g=1.0):
    n = noise(d); t = tt(d); out = np.zeros_like(n); seg = int(0.01 * SR)
    for i in range(0, len(n), seg):           # time-varying bandpass
        p = i / len(n); a, b = min(f0, 1600), min(f1, 1600); fc = a * (b / a) ** (np.sin(np.pi * p) if b > a else p)
        out[i:i + seg] = filt(n[max(0, i - 2000):i + seg], "band", [fc * 0.6, min(fc * 1.6, 20000)])[-len(n[i:i + seg]):]
    return filt(out, "low", 1400, 4) * np.sin(np.pi * t / d) ** 2 * g
def riser(d):
    t = tt(d); p = t / d
    n = whoosh(d, 200, 9000) * p ** 1.5 * 1.3
    tone = np.sin(2 * np.pi * np.cumsum(150 + 250 * p ** 2) / SR) * p ** 2 * 0.10
    return n + tone
def impact(g=1.0):
    t = tt(2.2)
    boom = np.sin(2 * np.pi * np.cumsum(38 + 70 * np.exp(-t * 10)) / SR) * np.exp(-t * 2.2)
    crash = filt(noise(2.2), "high", 2500) * np.exp(-t * 5.5) * 0.35
    return (np.tanh(boom * 2) * 0.55 + crash * 1.3) * g
def blip(f, d=0.18, g=1.0):
    return (np.sin(2 * np.pi * f * tt(d)) + 0.3 * np.sin(4 * np.pi * f * tt(d))) * np.exp(-tt(d) * 25) * g
def click():
    return filt(noise(0.02), "high", 2500) * np.exp(-tt(0.02) * 300)


# ---------------- arrangement ----------------
# chords per 2 s bar: Am F C G (vi IV I V), intro stays on Am
PROG = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
ROOT = [45, 41, 36, 43]

# intro 0-7: tense filtered 8th pulse + low drone, no kick
for k, b in enumerate(np.arange(0.0, DROP, BEAT / 2)):
    cut = 600 + 2600 * (b / DROP) ** 2                 # filter opens toward the drop
    s = filt(saw(mid(45 if k % 4 else 57), 0.2, 10), "low", cut) * np.exp(-tt(0.2) * 12)
    add("music", s, b, pan=0.25 * (-1) ** k, g=0.38)
add("music", filt(saw(mid(33), DROP, 8), "low", 220) * np.minimum(tt(DROP) / 0.6, 1), 0, g=0.10)
for b in np.arange(4.0, DROP, BEAT):                   # soft pre-drop pulse kick (muffled)
    add("drums", filt(kick(), "low", 300), b, g=0.22)
add("music", pad(PROG[0], 4.2), 3.0, g=0.10)

# main section DROP -> KICK_END
kicks = np.arange(DROP, KICK_END - 1e-6, BEAT)
for b in kicks:
    add("drums", kick(), b, g=0.55)
for b in np.arange(DROP + BEAT, KICK_END, 2 * BEAT):
    add("drums", clap(), b, g=0.6, pan=0.05)
for b in np.arange(DROP, KICK_END, BEAT / 2):
    off = abs(((b - DROP) / BEAT) % 1 - 0.5) < 1e-6
    add("drums", hat(open_=off), b, pan=-0.3 if off else 0.3, g=0.20 if off else 0.09)
bar = 0
for bs in np.arange(DROP, KICK_END, 4 * BEAT):
    ch, r = PROG[bar % 4], ROOT[bar % 4]
    add("music", pad([n + 12 for n in ch], 2.0 + 0.3), bs, g=0.24)
    for k in range(4):                                  # offbeat bass
        add("music", bass(mid(r)), bs + k * BEAT + BEAT / 2, g=0.24)
    arp = [ch[0] + 24, ch[1] + 24, ch[2] + 24, ch[1] + 24, ch[0] + 36, ch[2] + 24, ch[1] + 24, ch[2] + 12]
    if bs >= 9.0:
        for k in range(16):                             # 16th plucks
            add("music", pluck(mid(arp[k % 8])), bs + k * BEAT / 4, pan=0.45 * np.sin(k * 1.3), g=0.17)
    bar += 1
for m in [69, 72, 76, 81]:                              # drop stab
    add("music", pluck(mid(m), 0.6), DROP, g=0.25)

# sidechain pump on the music bus from the kick
sc = np.ones(N)
for b in kicks:
    i = int(b * SR); n = min(int(0.3 * SR), N - i)
    sc[i:i + n] = np.minimum(sc[i:i + n], 1 - 0.65 * np.exp(-np.arange(n) / SR / 0.09))
BUS["music"] *= sc[:, None]

# outro 26.3-30: big check impact, open chord, sparkle
for m in [45, 57, 64, 69, 72, 76]:
    add("music", pad([m], 3.6), 26.3, g=0.20)
for k, m in enumerate([81, 84, 88, 93, 88, 84]):
    add("music", pluck(mid(m), 0.5), 27.1 + k * BEAT / 2, pan=0.4 * (-1) ** k, g=0.10)

# ---------------- sound design ----------------
add("sfx", riser(0.5), 0.0, g=0.052)                      # opening swell
add("sfx", whoosh(0.55, 200, 3000), 0.15, pan=-0.4, g=0.075)   # card slides in
add("sfx", impact(0.7), 1.22)            # DECLINED stamp
add("sfx", filt(noise(0.08), "low", 1200) * np.exp(-tt(0.08) * 40), 1.22, g=0.6)
for k, s in enumerate([3.02, 3.52, 4.02]):               # kinetic words
    add("sfx", whoosh(0.4, 400, 7000), s - 0.05, pan=0.5 - 0.5 * k, g=0.083)
add("sfx", whoosh(0.5, 6000, 300), 4.7, pan=-0.3, g=0.068)    # words exit
for i, tk in enumerate(np.linspace(5.15, 6.25, 12)):     # counter ticks
    add("sfx", blip(1800 + i * 60, 0.05), tk, g=0.18)
add("sfx", blip(mid(81), 0.6, 0.4), 6.3)
add("sfx", riser(1.2), DROP - 1.2, g=0.12)                # build into the drop
add("sfx", impact(0.7), DROP)
add("sfx", whoosh(0.9, 300, 9000), 6.8, g=0.105)           # teal light sweep
add("sfx", blip(mid(88), 0.5, 0.35), 7.85)               # tagline pill
for s in [9.0, 13.0, 17.0, 21.0, 25.0]:                  # card transitions
    add("sfx", whoosh(0.55, 250, 6000), s - 0.25, pan=0.4, g=0.083)
add("sfx", blip(mid(84), 0.18, 0.5), 9.55)               # badge pop
add("sfx", blip(mid(76), 0.12, 0.25), 11.1)              # delta chip
for i in range(12):                                      # SHAP bars
    add("sfx", blip(mid(72 + (i % 5) * 2), 0.06, 0.12), 13.55 + i * 0.17, pan=0.3 * np.sin(i))
for c in [19.0, 25.6, 26.15]:
    add("sfx", click(), c, g=0.7)
add("sfx", blip(mid(79), 0.15, 0.3), 19.25); add("sfx", blip(mid(86), 0.25, 0.35), 19.35)  # Approve flip
for k in range(18):                                      # letter typing ticks
    add("sfx", click() * 0.25, 21.5 + k * 0.12 + 0.02 * np.sin(k * 3), pan=0.2)
add("sfx", blip(mid(88), 0.4, 0.4), 23.0); add("sfx", blip(mid(91), 0.4, 0.4), 23.55)
add("sfx", riser(0.6), 25.7, g=0.09)
add("sfx", impact(1.0), 26.3)                            # big green check
for i, m in enumerate([84, 88, 91, 96]):
    add("sfx", blip(mid(m), 0.5, 0.25), 26.45 + i * 0.06)
add("sfx", whoosh(0.6, 300, 5000), 26.9, g=0.06)          # into end card
add("sfx", blip(mid(93), 1.2, 0.15) , 27.85)             # URL pill shimmer

# ---------------- mix ----------------
def reverb(x, d=1.6, mix=0.18):
    ir = noise(d) * np.exp(-tt(d) * 4.5); ir = filt(ir, "low", 6000); ir /= np.sqrt((ir ** 2).sum())
    wet = np.stack([fftconvolve(x[:, c], np.roll(ir, c * 37))[:len(x)] for c in range(2)], 1)
    return x + wet * mix

music = reverb(BUS["music"], 1.8, 0.22)
sfx = reverb(BUS["sfx"], 1.2, 0.15)
mix = BUS["drums"] * 0.9 + music * 1.0 + sfx * 0.85
for c in range(2): mix[:, c] = filt(mix[:, c], "high", 30) + 0.12 * filt(mix[:, c], "high", 3500)
t = np.arange(N) / SR
mix *= np.minimum(t / 0.05, 1)[:, None] * np.clip((D - t) / 1.2, 0, 1)[:, None]
mix /= np.abs(mix).max()
mix = np.tanh(mix * 1.1) / np.tanh(1.1)          # gentle glue, no audible distortion
mix *= 0.8
with wave.open("build/music.wav", "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
rms = np.sqrt((mix ** 2).mean()); print("peak", np.abs(mix).max().round(3), "rms", rms.round(3))
