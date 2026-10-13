"use strict";
let folderTarget=null,folderCurrent=null,folderRequest=0;
$('openWorkspace').onclick=()=>{
  if(!S.config)return;
  if(S.busy || S.samRunning || (typeof exporting==='function' && exporting())) return toast('진행 중인 작업이 끝난 뒤 폴더를 바꾸세요.',true);
  $('workspaceInput').value=S.config?.dataset_root||'';
  $('workspaceOutput').value=S.config?.root||'';
  $('workspaceStatus').textContent='';$('workspaceDialog').showModal();
};
$('workspaceClose').onclick=()=>$('workspaceDialog').close();
$('workspaceApply').onclick=async()=>{
  if(!canLeave())return;
  const button=$('workspaceApply');button.disabled=true;$('workspaceStatus').textContent='이미지와 라벨을 준비하는 중…';
  try{
    const result=await api('/api/workspace',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':S.config.token},body:JSON.stringify({dataset_root:$('workspaceInput').value.trim(),output_root:$('workspaceOutput').value.trim()})});
    $('workspaceStatus').textContent=`${result.frames}장 준비 완료`;
    S.savedId=S.stateId;S.polygon=[];S.samProposal=null;S.drag=null;
    window.location.reload();
  }catch(error){$('workspaceStatus').textContent=error.message;button.disabled=false;}
};
async function browseFolders(path){
  const request=++folderRequest;
  folderCurrent=null;$('folderStatus').textContent='폴더를 불러오는 중…';$('folderList').replaceChildren();
  for(const id of ['folderPath','folderGo','folderUp','folderSelect'])$(id).disabled=true;
  try{
    const data=await api('/api/folders?path='+encodeURIComponent(path));
    if(request!==folderRequest)return;
    folderCurrent=data;$('folderPath').value=data.path;$('folderStatus').textContent='';
    for(const item of data.folders){const b=document.createElement('button');b.type='button';b.textContent='📁 '+item.name;b.onclick=()=>browseFolders(item.path);$('folderList').append(b);}
  }catch(error){if(request===folderRequest)$('folderStatus').textContent=error.message;}
  finally{if(request===folderRequest)for(const id of ['folderPath','folderGo','folderUp','folderSelect'])$(id).disabled=false;}
}
async function openFolder(target){folderTarget=target;$('folderDialog').showModal();await browseFolders($(target).value||S.config.dataset_root);}
$('browseInput').onclick=()=>openFolder('workspaceInput');$('browseOutput').onclick=()=>openFolder('workspaceOutput');
$('folderGo').onclick=()=>browseFolders($('folderPath').value);$('folderUp').onclick=()=>folderCurrent&&browseFolders(folderCurrent.parent);
$('folderClose').onclick=()=>$('folderDialog').close();$('folderSelect').onclick=()=>{if(folderCurrent&&folderTarget){$(folderTarget).value=folderCurrent.path;$('folderDialog').close();}};
