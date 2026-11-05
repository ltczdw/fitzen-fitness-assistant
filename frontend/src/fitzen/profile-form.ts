import type {Profile} from './state'
export type ProfileForm={display_name:string;goal:string;experience:string;available_days:number|string;height_cm:number|string|null;body_weight_kg:number|string|null;target_weight_kg:number|string|null;body_fat_pct:number|string|null}
export function profileForm(profile:Profile):ProfileForm {
 return {display_name:profile.display_name||'我的训练',goal:profile.goal||'保持健康',experience:profile.experience||'初学者',available_days:profile.available_days??3,height_cm:profile.height_cm??null,body_weight_kg:profile.body_weight_kg??null,target_weight_kg:profile.target_weight_kg??null,body_fat_pct:profile.body_fat_pct??null}
}
export function profilePayload(form:ProfileForm):Profile {
 const display_name=form.display_name.trim(),goal=form.goal.trim(),experience=form.experience.trim()
 if(!display_name||display_name.length>100)throw new Error('称呼请填写1到100个字')
 if(!goal||goal.length>100)throw new Error('训练目标请填写1到100个字')
 if(!experience||experience.length>100)throw new Error('训练经验请填写1到100个字')
 const available_days=Number(form.available_days)
 if(!Number.isInteger(available_days)||available_days<1||available_days>7)throw new Error('每周训练天数请填写1到7之间的整数')
 const payload:Profile={display_name,goal,experience,available_days}
 const fields=[['height_cm','身高',100,250],['body_weight_kg','体重',20,400],['target_weight_kg','目标体重',20,400],['body_fat_pct','体脂率',0,100]] as const
 for(const [key,label,min,max] of fields){
  const raw=form[key]
  if(raw===''||raw==null){payload[key]=null;continue}
  const value=Number(raw)
  if(!Number.isFinite(value)||value<min||value>max||(key==='body_fat_pct'&&(value===0||value===100)))throw new Error(key==='body_fat_pct'?'体脂率请填写大于0、小于100的百分比':`${label}请填写${min}到${max}之间的数值`)
  payload[key]=value
 }
 return payload
}
