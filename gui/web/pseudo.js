"use strict";
const PJ={job:null,sending:false,polling:false,error:""};
function pseudoActive(){return PJ.job && ['queued','running','cancelling'].includes(PJ.job.status);}
function pseudoModel(){return S.config.pseudo.models.find(m=>m.id===$('pseudoModel').value);}
function renderPseudo(){
  const active=Boolean(pseudoActive()||PJ.sending), multi=$('pseudoModel').value==='multi_task';
  for(const id of ['pseudoModel','pseudoCheckpoint','pseudoTemplate','pseudoPython','pseudoOutput','pseudoDevice','pseudoConf','pseudoLimit','pseudoMapping','pseudoHeight','pseudoWidth','pseudoBatch','pseudoBrowse','pseudoBrowseTemplate'])$(id).disabled=active;
  $('pseudoTemplateField').hidden=$('pseudoTemplateHelp').hidden=!multi;
  $('pseudoMappingField').hidden=multi;
  $('pseudoBatchField').hidden=!multi;
  $('pseudoStart').disabled=active;
  $('pseudoCancel').disabled=!pseudoActive();
  $('pseudoOpenResult').disabled=active || PJ.job?.status!=='completed';
  if(PJ.error)$('pseudoStatus').textContent=PJ.error;
  const model=pseudoModel();
  $('pseudoModelHelp').textContent=model ? (model.nir?'RGB와 대응 NIR이 필요합니다. 기존 공동 모델의 클래스 구성을 사용합니다.':'RGB만 사용합니다. 클래스 이름이 일치하면 자동 연결하며, 다른 클래스는 아래 JSON 매핑으로 지정합니다.') : '';
  $('pseudoProgress').hidden=!PJ.job;
  if(PJ.job){
    const job=PJ.job;
    if(job.total){$('pseudoProgress').max=job.total;$('pseudoProgress').value=job.done;}else $('pseudoProgress').removeAttribute('value');
    const names={queued:'실행 대기',running:'생성 중',cancelling:'중단 중',completed:'생성 완료',cancelled:'중단됨',interrupted:'서버 재시작으로 중단',error:'생성 실패'};
    if(!PJ.error)$('pseudoStatus').textContent=`${names[job.status]||job.status} · ${job.done}/${job.total}장\n${job.output}${job.error?'\n'+job.error:''}`;
    $('pseudoLog').textContent=job.log||'';
    $('pseudoLog').scrollTop=$('pseudoLog').scrollHeight;
  }
}
function defaultCheckpoint(){
  const model=pseudoModel();
  if(!model)return;
  $('pseudoCheckpoint').value=model.checkpoints.find(p=>!p.endsWith('yolo26s.pt'))||'';
  $('pseudoTemplate').value=model.checkpoints.find(p=>p.endsWith('yolo26s.pt'))||'';
  $('pseudoWeights').replaceChildren();
  model.checkpoints.forEach(p=>$('pseudoWeights').append(new Option(p,p)));
  $('pseudoMapping').value='{}';renderPseudo();
}
async function pollPseudo(){
  if(PJ.polling||PJ.sending)return;
  PJ.polling=true;
  try{PJ.job=(await api('/api/pseudo-labels/latest')).job;renderPseudo();}
  catch(error){$('pseudoStatus').textContent=error.message;}
  finally{PJ.polling=false;}
}
$('pseudoLabeling').onclick=async()=>{
  if(!S.config||S.busy)return;
  S.config.pseudo=(await api('/api/config')).pseudo;
  const previous=$('pseudoModel').value;
  $('pseudoModel').replaceChildren();
  S.config.pseudo.models.forEach(m=>$('pseudoModel').append(new Option(m.name,m.id)));
  if(previous)$('pseudoModel').value=previous;
  if(!$('pseudoPython').value)$('pseudoPython').value=S.config.pseudo.python;
  if(!$('pseudoOutput').value){
    const stamp=new Date().toISOString().replace(/[^0-9]/g,'').slice(0,14);
    $('pseudoOutput').value=S.config.pseudo.output_parent+'/pseudo_'+stamp;
  }
  if(!$('pseudoCheckpoint').value)defaultCheckpoint();
  $('pseudoDialog').showModal();await pollPseudo();renderPseudo();
};
$('pseudoClose').onclick=()=>$('pseudoDialog').close();
$('pseudoModel').onchange=defaultCheckpoint;
$('pseudoStart').onclick=async()=>{
  if(!canLeave())return;
  PJ.error="";PJ.sending=true;renderPseudo();
  try{
    const payload={task:$('pseudoModel').value,checkpoint:$('pseudoCheckpoint').value.trim(),model_template:$('pseudoTemplate').value.trim(),python:$('pseudoPython').value.trim(),output_root:$('pseudoOutput').value.trim(),device:$('pseudoDevice').value.trim(),conf:Number($('pseudoConf').value),limit:Number($('pseudoLimit').value),height:Number($('pseudoHeight').value),width:Number($('pseudoWidth').value),batch:Number($('pseudoBatch').value),class_mapping:$('pseudoModel').value==='multi_task'?{}:JSON.parse($('pseudoMapping').value)};
    PJ.job=(await api('/api/pseudo-labels',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':S.config.token},body:JSON.stringify(payload)})).job;
  }catch(error){PJ.error=error.message;toast(error.message,true);}
  finally{PJ.sending=false;renderPseudo();}
};
$('pseudoCancel').onclick=async()=>{
  try{PJ.job=(await api('/api/pseudo-labels/cancel',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':S.config.token},body:'{}'})).job;renderPseudo();}
  catch(error){toast(error.message,true);}
};
$('pseudoOpenResult').onclick=async()=>{
  if(!PJ.job||PJ.job.status!=='completed'||!canLeave())return;
  try{
    await api('/api/workspace',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':S.config.token},body:JSON.stringify({dataset_root:PJ.job.input,output_root:PJ.job.output})});
    S.savedId=S.stateId;S.polygon=[];S.samProposal=null;window.location.reload();
  }catch(error){toast(error.message,true);}
};
let pseudoFileTarget='pseudoCheckpoint', pseudoFileRequest=0;
async function browsePseudoFiles(path){
  const request=++pseudoFileRequest;
  $('pseudoFileList').replaceChildren();$('pseudoFileStatus').textContent='불러오는 중…';
  try{
    const data=await api('/api/pseudo-files?path='+encodeURIComponent(path));
    if(request!==pseudoFileRequest)return;
    $('pseudoFilePath').value=data.path;
    $('pseudoFileUp').onclick=()=>browsePseudoFiles(data.parent);
    for(const folder of data.folders){const b=document.createElement('button');b.textContent='📁 '+folder.name;b.onclick=()=>browsePseudoFiles(folder.path);$('pseudoFileList').append(b);}
    for(const file of data.files){const b=document.createElement('button');b.textContent=file.name;b.onclick=()=>{$(pseudoFileTarget).value=file.path;$('pseudoFileDialog').close();};$('pseudoFileList').append(b);}
    $('pseudoFileStatus').textContent=data.files.length?'체크포인트를 클릭해서 선택하세요.':'이 폴더에는 .pt 체크포인트가 없습니다. 경로를 이동하거나 파일을 넣으세요.';
  }catch(error){if(request===pseudoFileRequest)$('pseudoFileStatus').textContent=error.message;}
}
for(const [button,target] of [['pseudoBrowse','pseudoCheckpoint'],['pseudoBrowseTemplate','pseudoTemplate']])$(button).onclick=()=>{pseudoFileTarget=target;$('pseudoFileDialog').showModal();browsePseudoFiles(pseudoModel().checkpoint_dir);};
$('pseudoFileClose').onclick=()=>$('pseudoFileDialog').close();
$('pseudoFileGo').onclick=()=>browsePseudoFiles($('pseudoFilePath').value);
setInterval(()=>{if($('pseudoDialog').open||pseudoActive())pollPseudo();},1500);
