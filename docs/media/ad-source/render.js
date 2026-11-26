const {chromium}=require('playwright');const path=require('path');const {spawn}=require('child_process');
const FPS=30,DUR=30,N=FPS*DUR;
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1920,height:1080}});
await p.goto('file://'+path.resolve('build/ad.html'));await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(500);
const ff=spawn('ffmpeg',['-y','-f','image2pipe','-framerate',String(FPS),'-c:v','png','-i','-','-i','build/music.wav',
 '-c:v','libx264','-preset','slow','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-shortest','-movflags','+faststart','out/secondlook_ad_30s.mp4'],{stdio:['pipe','ignore','inherit']});
for(let i=0;i<N;i++){await p.evaluate(t=>render(t),i/FPS);const buf=await p.screenshot({type:'png'});
 if(!ff.stdin.write(buf))await new Promise(r=>ff.stdin.once('drain',r)); if(i%150==0)console.error('frame',i);}
ff.stdin.end();await new Promise(r=>ff.on('close',r));await b.close();})();
