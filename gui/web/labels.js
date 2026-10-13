"use strict";
let labelDraft=null;
function renderLabelTargets(){
  for(const [kind,id] of [['detection','detectionTargets'],['segmentation','segmentationTargets']]){
    $(id).replaceChildren();
    labelDraft[kind].forEach((item,index)=>{
      const row=document.createElement('div');row.dataset.labelId=item.id===null?'new':String(item.id);row.className='label-target-row'+(item.active?'':' inactive');
      const number=document.createElement('span');number.className='label-target-id';number.textContent=item.id===null ? '새 ID' : `ID ${item.id}`;
      const name=document.createElement('input');name.type='text';name.value=item.name;name.maxLength=80;name.setAttribute('aria-label',`${kind} 클래스 이름`);name.oninput=()=>item.name=name.value;
      const color=document.createElement('input');color.type='color';color.value=item.color;color.setAttribute('aria-label',`${item.name} 색상`);color.oninput=()=>item.color=color.value;
      const remove=document.createElement('button');remove.type='button';remove.textContent=item.active?'제거':'복원';
      remove.disabled=kind==='segmentation' && [0,255].includes(item.id);
      if(remove.disabled)remove.title='배경 0과 ignore 255는 필수 값입니다.';
      remove.onclick=()=>{
        if(item.active && item.id!==null && !window.confirm('이 클래스를 새 레이블링 대상에서 제외할까요? 기존 박스·마스크와 ID는 유지되며 나중에 복원할 수 있습니다.'))return;
        if(item.id===null)labelDraft[kind].splice(index,1);else item.active=!item.active;
        renderLabelTargets();
      };
      row.append(number,name,color,remove);$(id).append(row);
    });
  }
}
$('openLabelSettings').onclick=async()=>{
  if(!S.config || S.busy || S.samRunning || exporting())return toast('진행 중인 작업이 끝난 뒤 설정을 변경하세요.',true);
  try{
    const config=await api('/api/config');
    labelDraft={...structuredClone(config.label_schema),revision:config.schema_revision};
    $('labelSettingsStatus').textContent='';renderLabelTargets();$('labelSettingsDialog').showModal();
  }catch(error){toast(error.message,true);}
};
$('labelSettingsClose').onclick=()=>$('labelSettingsDialog').close();
for(const [kind,button] of [['detection','addDetectionTarget'],['segmentation','addSegmentationTarget']])$(button).onclick=()=>{
  labelDraft[kind].push({id:null,name:'',color:kind==='detection'?'#6dcfff':'#bd7cff',active:true});renderLabelTargets();
  $(kind==='detection'?'detectionTargets':'segmentationTargets').lastElementChild.querySelector('input[type=text]').focus();
};
$('labelSettingsSave').onclick=async()=>{
  if(dirty() || S.drag || S.polygon.length || S.samProposal){$('labelSettingsStatus').textContent='현재 프레임 편집을 먼저 저장하거나 취소한 후 설정을 적용하세요.';return;}
  const button=$('labelSettingsSave');button.disabled=true;
  try{
    const config=await api('/api/label-schema',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':S.config.token},body:JSON.stringify(labelDraft)});
    installLabelSchema({...S.config,...config});
    $('labelSettingsDialog').close();
    if(S.frame)await openFrame(S.frame.id,true);
    else updateState();
    toast('클래스와 색상을 저장했습니다. 기존 라벨과 ID는 유지됩니다.');
  }catch(error){$('labelSettingsStatus').textContent=error.message;}
  finally{button.disabled=false;}
};
