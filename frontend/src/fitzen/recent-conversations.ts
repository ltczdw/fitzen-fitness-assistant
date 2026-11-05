type Conversation = {conversation_id:string;title?:string;[key:string]:unknown}
type Message = {role:string;content:string}

// The active transcript is already visible in the chat. Derive its sidebar row
// immediately without claiming the server has persisted it. Switching to an
// empty chat removes this projection; the server remains the history source.
export function recentConversations(rows:Conversation[],currentId:string,messages:Message[],query='') {
 const firstMessage=messages.find(message=>message.role==='user'&&message.content.trim())
 const current=rows.find(row=>row.conversation_id===currentId)
 const list=currentId&&firstMessage
  ? [{...current,conversation_id:currentId,title:current?.title||firstMessage.content.slice(0,80)},...rows.filter(row=>row.conversation_id!==currentId)]
  : rows
 const term=query.toLocaleLowerCase()
 return list.filter(row=>String(row.title||'未命名对话').toLocaleLowerCase().includes(term)).slice(0,30)
}
