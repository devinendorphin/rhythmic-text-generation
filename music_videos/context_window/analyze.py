import numpy as np, scipy.io.wavfile as w, scipy.signal as s
sr,x=w.read('song.wav'); x=x.astype(np.float32).mean(1)/32768
FPS=30; hop=sr//FPS; n=len(x)//hop
fr=x[:n*hop].reshape(n,hop)
rms=np.sqrt((fr**2).mean(1))
# STFT bands
f,t,Z=s.stft(x,sr,nperseg=4096,noverlap=4096-hop)
M=np.abs(Z)[:, :n]
def band(lo,hi): return M[(f>=lo)&(f<hi)].mean(0)
bass=band(30,150); mid=band(300,2500); hi=band(4000,12000)
flux=np.maximum(np.diff(np.log1p(M*50),axis=1),0).sum(0); flux=np.r_[0,flux]
np.savez('feat.npz',rms=rms,bass=bass,mid=mid,hi=hi,flux=flux,fps=FPS)
# tempo via autocorr of flux
fl=flux-flux.mean(); ac=np.correlate(fl,fl,'full')[len(fl)-1:]
lags=np.arange(len(ac)); bpm=60*FPS/lags[1:]
m=(bpm>60)&(bpm<180); L=lags[1:][m][np.argmax(ac[1:][m])]
print('bpm~',60*FPS/L)
# per-2s energy profile
sec=rms[:n//(2*FPS)*2*FPS].reshape(-1,2*FPS).mean(1)
for i,v in enumerate(sec): print(f"{i*2:4d}s {'#'*int(v*200)}")
