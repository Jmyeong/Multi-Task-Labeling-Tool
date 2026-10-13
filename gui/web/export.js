"use strict";
const E = {job: null, sending: false, polling: false, error: ""};
function exporting() {return E.job && ["queued","snapshot","planning","copying"].includes(E.job.status);}
function exportDirty() {return dirty() || S.drag || S.polygon.length || S.samProposal;}
function renderExport() {
  const active = exporting() || E.sending;
  $("exportParent").disabled = $("exportName").disabled = Boolean(active);
  $("exportStart").disabled = Boolean(active || S.busy || exportDirty() || !$("exportParent").value.trim() || !$("exportName").value.trim());
  $("exportTarget").textContent = `출력 위치: ${$("exportParent").value.replace(/\/+$/,"")}/${$("exportName").value}`;
  if (E.error) {
    $("exportProgress").hidden = true;
    $("exportStatus").textContent = E.error;
    $("exportCurrentFile").textContent = "";
    return;
  }
  if (!E.job) {
    $("exportProgress").hidden = true;
    $("exportStatus").textContent = exportDirty() ? "현재 편집을 먼저 저장하거나 취소하세요." : "승인된 결과를 복사할 위치를 입력하세요.";
    $("exportCurrentFile").textContent = "";
    return;
  }
  const job = E.job;
  $("exportProgress").hidden = false;
  if (["queued","snapshot","planning"].includes(job.status)) $("exportProgress").removeAttribute("value");
  else {
    $("exportProgress").max = Math.max(1,job.bytes_total);
    $("exportProgress").value = job.bytes_done;
  }
  const labels = {queued:"추출 대기",snapshot:"승인된 라벨 확인 중",planning:"파일·타임스탬프 확인 중",copying:"복사 중",completed:"추출 완료",error:"추출 실패"};
  const percent = job.bytes_total ? Math.floor(job.bytes_done/job.bytes_total*100) : 0;
  $("exportStatus").textContent = job.status === "error" ? `추출 실패: ${job.error}` : job.status === "completed"
    ? `${job.frames}개 승인 프레임 추출 완료\n${job.output}`
    : `${labels[job.status] || job.status} · ${job.frames}장 · ${job.files_done}/${job.files_total}개 파일 · ${percent}%`;
  $("exportCurrentFile").textContent = job.current_file || "";
}
async function pollExport() {
  if (E.polling || E.sending) return;
  E.polling = true;
  try {
    E.job = (await api("/api/dataset-exports/latest")).job;
    renderExport();
  } catch (error) {$("exportStatus").textContent=error.message;}
  finally {E.polling=false;}
}
$("exportDataset").onclick = async () => {
  if (!S.config) return;
  if (!$("exportParent").value) $("exportParent").value=S.config.export_parent;
  if (!$("exportName").value) {
    const now=new Date(),part=n=>String(n).padStart(2,"0");
    $("exportName").value=`reviewed_${now.getFullYear()}${part(now.getMonth()+1)}${part(now.getDate())}_${part(now.getHours())}${part(now.getMinutes())}${part(now.getSeconds())}`;
  }
  $("datasetExportDialog").showModal();renderExport();await pollExport();
};
$("exportClose").onclick=()=>$("datasetExportDialog").close();
$("exportParent").oninput=$("exportName").oninput=()=>{E.error="";renderExport();};
$("exportStart").onclick=async()=>{
  if ($("exportStart").disabled) return;
  E.error="";E.sending=true;renderExport();
  try {
    const result=await api("/api/dataset-exports",{method:"POST",headers:{"Content-Type":"application/json","X-Review-Token":S.config.token},
      body:JSON.stringify({parent:$("exportParent").value,name:$("exportName").value})});
    E.job=result.job;
  } catch(error) {E.error=error.message;}
  finally {E.sending=false;renderExport();}
};
setInterval(()=>{if($("datasetExportDialog").open || exporting()) pollExport();},1500);
