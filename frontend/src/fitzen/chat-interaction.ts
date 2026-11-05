// Suppress only the key event that commits an IME candidate. A later key
// press must not be swallowed by a fixed post-composition timeout.
export function createCompositionGuard(schedule:(done:()=>void)=>ReturnType<typeof setTimeout>=done=>setTimeout(done,0)){
 let composing=false,commitPending=false,timer:ReturnType<typeof setTimeout>|undefined
 function release(){commitPending=false;clearTimeout(timer);timer=undefined}
 function end(){composing=false;release();commitPending=true;timer=schedule(release)}
 return {
  start(){release();composing=true},
  end,
  cancel(){composing=false;release()},
  active(event:{isComposing?:boolean;keyCode?:number}){return composing||event.isComposing===true||event.keyCode===229},
  blocksCommitEnter(){return commitPending},
  keyup(event:{key:string}){if(event.key==='Enter')release()},
  dispose(){composing=false;release()},
 }
}
export type ChatScrollMetrics={scrollHeight:number;scrollTop:number;clientHeight:number}
export function createChatScrollPolicy(){
 let following=true,lastTop=0
 const distance=(el:ChatScrollMetrics)=>Math.max(0,el.scrollHeight-el.clientHeight-el.scrollTop)
 return {
  get following(){return following},
  pause(el:ChatScrollMetrics){following=false;lastTop=el.scrollTop},
  reset(){following=true;lastTop=0},
  track(el:ChatScrollMetrics,line:number){
   const down=el.scrollTop>lastTop,d=distance(el)
   if(d>=line)following=false
   else if(down)following=true
   lastTop=el.scrollTop
  },
  followed(el:ChatScrollMetrics){lastTop=el.scrollTop},
  showLatest(el:ChatScrollMetrics,line:number){return distance(el)>3*line},
 }
}
