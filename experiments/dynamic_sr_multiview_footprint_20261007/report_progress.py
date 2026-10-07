"""Refresh progress from frozen actual receipts, without GPU, SSH or publication.

Run using the bundled documents Python after its operation marker, then render
and inspect every page. Complete prefixes require both complete and exact exit.
"""
from __future__ import annotations
import copy
import hashlib
import json
import time
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from fp_common import ROOT,OUT,module,Path

STEM=ROOT/'docs/dynamic_sr_multiview_footprint_progress_2026-10-07'
PREFIX=OUT/'full_LR_prefixes/cook_spinach/seed_20261007'


def sha(raw):return hashlib.sha256(raw).hexdigest()

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def accepted_exit0(terminal):
    return bool(terminal.get('exit_proved') and terminal.get('successful') and terminal.get('exit_code')==0)


def verify_b0_chain(receipt,segments,terminal):
    """Count closed formal work only, with a contiguous immutable segment chain."""
    if receipt.get('status')!='completed_training' or not accepted_exit0(terminal):
        raise ValueError('B0 completion and exact Exit0 both required')
    cursor=0
    for segment in segments:
        if segment['start']!=cursor or segment['updates']!=segment['suffix_endpoint']-cursor:
            raise ValueError('B0 segment gap or duplicate formal work')
        cursor=segment['suffix_endpoint']
    if cursor!=receipt['suffix_endpoint'] or sum(s['updates'] for s in segments)!=receipt['updates']:
        raise ValueError('B0 completion total disagrees with segments')
    for key in ('training_rgb_forwards','adam_calls'):
        if sum(s[key] for s in segments)!=receipt[key]:raise ValueError('B0 cost total disagreement')
    return cursor


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_{time.time_ns()}'
    frozen=OUT/'progress_report_frozen_inputs'/stamp
    frozen.mkdir(parents=True,exist_ok=False)
    archive=OUT/'progress_report_archive'/stamp
    archive.mkdir(parents=True,exist_ok=False)
    identities={}; captured={}
    def read(path,required=True):
        path=Path(path)
        key=str(path.relative_to(ROOT))
        if key in captured:return captured[key]
        if not path.exists():
            if required:raise FileNotFoundError(path)
            identities[key]=dict(status='absent_at_snapshot');return None
        raw=path.read_bytes();dst=frozen/'inputs'/key
        dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
        identities[key]=dict(sha256=sha(raw),bytes=len(raw),frozen=str(dst.relative_to(ROOT)))
        captured[key]=json.loads(raw)
        return captured[key]
    def bound_json(ref):
        path=ROOT/ref['path'];value=read(path)
        if identities[str(path.relative_to(ROOT))]['sha256']!=ref['sha256']:
            raise ValueError('Receipt binding changed: '+ref['path'])
        return value
    for suffix in ('.docx','.md','.codex.md'):
        source=STEM.with_suffix(suffix)
        if source.exists():(archive/source.name).write_bytes(source.read_bytes())
    for name in ('progress_report_snapshot.json','progress_report_QA.json'):
        source=OUT/name
        if source.exists():(archive/name).write_bytes(source.read_bytes())
    reference=archive/STEM.with_suffix('.docx').name
    if not reference.exists():raise ValueError('Retained Word reference required')
    state=read(OUT/'state.json');protocol=read(OUT/'protocol.json')
    development=read(OUT/'full_data_readiness/full_development_ready_manifest_index.json')
    read(OUT/'full_data_readiness/full_prefix_readiness.json')
    native=bound_json(state['native_fixture'])
    parent=read(OUT/'preparation/CPU_to_GPU_interruption_20261007/index.json')
    stop=read(OUT/'preparation/CPU_to_GPU_interruption_20261007/stop_receipt.json')
    b0exit=read(OUT/'completion/B0_curve_exit_acceptance.json')
    b0first=read(OUT/'runs/r1_B0/attempt_complete_0000_0100.json')
    b0curve=bound_json(b0exit['segment'])
    assert b0first['start']==0 and b0first['suffix_endpoint']==b0curve['start']==100
    assert b0curve['suffix_endpoint']==3000 and b0curve['updates']==2900
    b0endpoint=bound_json(state['B0_endpoint'])
    b0complete=bound_json(b0endpoint['training'])
    b0segments=[bound_json(ref) for ref in b0complete['segments']]
    b0updates=verify_b0_chain(b0complete,b0segments,b0endpoint['terminal'])
    formal=state['formal_updates_completed'];endpoints=state['full_endpoints_completed']
    training_endpoints=state['training_endpoints_completed']
    assert formal>=b0updates and training_endpoints>=1 and endpoints<=training_endpoints
    coarse=read(OUT/'completion/full_prefix_RNGfix_exit_acceptance.json')
    term=coarse['terminal']
    assert term['exit_proved'] and term['successful'] and term['exit_code']==0
    coarsecp=read(PREFIX/'checkpoint_coarse_03000.json')
    assert coarsecp['sha256']==coarse['checkpoint']['sha256']
    fine=read(OUT/'completion/full_prefix_fine14000_registration.json')
    read(PREFIX/'resume_amendments/RNG_CPU_transport_ab5fd0313d37bbfa.json')
    prefixcomplete=read(PREFIX/'complete.json',False)
    fineexit=read(OUT/'completion/full_prefix_fine14000_exit_acceptance.json',False)
    fine_exit_registrations=[]
    for path in sorted((OUT/'completion').glob('full_prefix*fine*registration.json')):
        fine_exit_registrations.append(read(path))
    finepassed=bool(prefixcomplete and prefixcomplete.get('status')=='completed_independent_full_author_LR_prefix' and fineexit and fineexit.get('terminal',{}).get('invocation_id')==fine['invocation_id'] and fineexit['terminal'].get('exit_proved') and fineexit['terminal'].get('successful') and fineexit['terminal'].get('exit_code')==0)
    if finepassed:
        assert bound_json(fineexit['complete'])==prefixcomplete
        assert fineexit['actual_state']==prefixcomplete['state']
        assert prefixcomplete['state']['coarse_iteration']==3000 and prefixcomplete['state']['fine_iteration']==14000
        assert prefixcomplete['state']['accepted_RGB']==34000 and prefixcomplete['state']['accepted_backward']==17000
        assert prefixcomplete['state']['accepted_Adam']==16999 and prefixcomplete['state']['optimizer_resets']==2
        assert prefixcomplete['checkpoint']==fineexit['checkpoint']
    reuse=read(OUT/'full_teacher_prepare/cook_spinach/reuse_copy_complete.json')
    read(OUT/'full_teacher_prepare/cook_spinach/reuse_copy_identity_verification.json')
    t60exit=read(OUT/'completion/full_teacher_first60_exit_acceptance.json')
    t300exit=read(OUT/'completion/full_teacher_through300_exit_acceptance.json')
    t60=bound_json(t60exit['receipt']);t300=bound_json(t300exit['batch'])
    t900=read(OUT/'completion/full_teacher_through900_registration.json')
    t900exit=read(OUT/'completion/full_teacher_through900_exit_acceptance.json',False)
    t900batch=bound_json(t900exit.get('batch',t900exit.get('receipt'))) if t900exit else None
    if t900exit:
        tterm=t900exit['terminal']
        assert tterm['invocation_id']==t900['invocation_id'] and tterm['exit_proved'] and tterm['successful'] and tterm['exit_code']==0
    evalreg=read(OUT/'completion/registration.json')
    preparereg=read(OUT/'completion/bindings/prepare_67185cbf201046cf.json')
    read(OUT/'completion/prepare_return_connection.json',False)
    advconfig=read(OUT/'advance/root_registration_v1.json')
    advreg=read(OUT/'advance/service_registration.json')
    bound_json(advreg['config'])
    read(OUT/'advance/state.json')
    read(OUT/'full_data_readiness/confirmation_failure_detected_CPU.json')
    failure_root=OUT/'full_data_readiness/incidents/official_range_failure_20261007_v1/incidents'
    for path in sorted(failure_root.glob('download_failure_*/failure.json')):read(path)
    localcal=read(OUT/'calibration.json',False)
    localprepare=read(OUT/'preparation/complete.json',False)
    read(OUT/'completion/prepare_return_receipt.json',False)
    remote=bound_json(state['remote_preparation'])
    assert remote['status']=='closed_remote_preparation_metadata_SHA_verified'
    assert remote['preparation']['status']=='completed_native_support_single_calibration'
    assert accepted_exit0(remote['terminal']) and remote['formal_updates']==0
    cal=remote['calibration'];calpassed=cal.get('status')=='passed'
    assert calpassed and cal['parent_state_unchanged'] and cal['cost']['parameter_updates']==0 and cal['cost']['Adam_calls']==0
    dispatch=bound_json(state['remote_paired_dispatch'])
    assert bound_json(dispatch['remote_preparation'])==remote
    assert bound_json(dispatch['uniform_evaluator'])==evalreg
    worker=dispatch['worker']
    r2failure=read(OUT/'advance/incidents/remote_log_directory_pre_python_failure_20261007/exact_exit_and_zero_work.json')
    assert r2failure['terminal']['exit_proved'] and r2failure['terminal']['exit_code']!=0
    assert r2failure['formal_updates']==0 and r2failure['GPU_calls']==0
    assert worker['invocation_id']!=r2failure['terminal']['invocation_id']
    gpu0=read(OUT/'resource_GPU0_decision_20261007.json')
    assert gpu0['status']=='user_decision_wait_existing_openpi_services_exit_no_shared_training'
    refinement=read(OUT/'operator_checks/full_refine/delivery.json')
    refinecheck=bound_json(refinement['final_CPU_receipt'])
    supportcheck=read(OUT/'operator_checks/full_support/1791384046929342571/receipt.json')
    evaluatecheck=read(OUT/'operator_checks/full_evaluate/1791383797703950404/receipt.json')
    assert refinecheck['status']=='passed_CPU_native_full_refinement_contracts'
    assert supportcheck['status']=='passed_CPU_contract_requires_separate_native_GPU_acceptance'
    assert evaluatecheck['status']=='passed_full_native_evaluation_CPU_contracts'
    assert not evaluatecheck['cuda_initialized_before'] and not evaluatecheck['cuda_initialized_after']
    assert evaluatecheck['native_raster_calls']==0 and evaluatecheck['GPU_forwards']==0 and evaluatecheck['Adam_calls']==0
    # A remote metadata proof never substitutes for the immutable local return.
    localaccepted=bool(localprepare and localprepare.get('status')=='completed_native_support_single_calibration' and localcal and localcal.get('status')=='passed')
    if localaccepted:raise ValueError('Local preparation changed; refresh report transfer narrative from its acceptance receipt')
    assert native['status']=='passed_native_CUDA_X_gradient_fixture'
    assert parent['status']=='completed_HR_parent_moments' and len(parent['entries'])==1140
    assert preparereg['event']['reconnected_existing_watcher'] is True
    assert preparereg['invocation_id']==preparereg['event']['invocation_id']
    at=datetime.now(timezone.utc).isoformat()
    blocks=[]
    def p(text):blocks.append(dict(type='paragraph',text=text))
    def h(text):blocks.append(dict(type='heading',text=text,level=1))
    def table(headers,rows,widths):blocks.append(dict(type='table',headers=headers,rows=rows,widths=widths))
    def page():blocks.append(dict(type='page'))
    blocks.append(dict(type='title',text='同刻多视图与像素足迹训练进度'))
    p(datetime.fromisoformat(at).astimezone(ZoneInfo('America/Los_Angeles')).strftime('%Y年%m月%d日 %H时%M分  America Los Angeles  本地回执快照'))
    p(f'正式短窗已接受至少 {formal} / 72000 更新。6000 步训练端点已完成 {training_endpoints} / 12；统一质量评价完成 {endpoints} / 12。B0 r1 的完整训练端点已有精确退出码 0 及检查点 SHA 验收。远端准备和一次校准已通过，本地整包回传未验收；r2 已恢复持久派发，运行中更新不计为完成。尚无 M/X 有效性或最终质量结论。')
    p(f'B0 r1 在本机 PRO6000 连续接受 0→100→3000→6000，累计训练 {b0complete["train_s"]:.3f} 秒、段墙钟 {b0complete["wall_s"]:.3f} 秒、峰值显存 {b0complete["peak_gb"]:.3f} GB。实际 {b0complete["training_rgb_forwards"]} RGB 前向、{b0complete["adam_calls"]} Adam 调用。三段 SHA 逐一核对；旧 3000 步服务退出未知的历史记录仍保留，当前 6000 步端点有独立精确退出证据。')
    p('低分辨率记为 LR，高分辨率记为 HR，超分辨率记为 SR。共享动态表示可能把某视角的教师细节带到其他视角。本轮分别检验同刻多图真实 LR 监督 M，以及把源视图投影并降采样后解释目标相机 LR 的像素足迹约束 X。完整 SR 教师监督保留，上一轮 R/G 结果只作为历史来源。')
    h('六臂实验及公平比较')
    table(['实验臂','相机时间与真实 LR 监督','教师与跨视图约束'],[
        ['B0','原随机三列；仅锚点 a','b/c 完整教师各 0.05'],
        ['Bsync','同刻三相机；仅锚点 a','与 B0 相同'],
        ['M','同刻三相机；三份 LR 平均','与 B0 相同'],
        ['X','同刻三相机；仅锚点 a','完整教师及六条有向 X'],
        ['MX','同刻三相机；三份 LR 平均','完整教师及六条有向 X'],
        ['E','原随机三列；锚点 L1/MSE 混合','与 B0 相同；训练数据一次校准'],
    ],[1.8,8.0,8.2])
    p('每臂两套后缀，各追加 6000 步；共同 U6000 父状态、132972 点、两套 Adam 优化器、随机状态、学习率和固定拓扑均继承。每步三个 RGB 前向后一次总反向，两优化器各更新一次。Adam 利用梯度历史调整步幅，因此续接必须保留它的状态。每套六臂使用相同训练平台。')
    budget=protocol['budget']
    p(f'固定总预算为 {budget["core_updates"]} 更新、{budget["core_RGB"]} RGB 前向、{budget["core_moments"]} 辅助矩前向及 {budget["adam_calls"]} Adam 调用。B0 与 X 首 100 步均属于正式预算。数据准备、校准、诊断和失败成本另计；相同更新数不等于相同计算成本。')
    page()
    h('原生验收及冻结支持准备')
    p('像素足迹算子通过 13 项 CPU 检查。它先把源 HR RGB 按当前目标轴向深度投影，再执行原 bicubic 抗混叠降采样和 0–1 限幅。缩小时采用可微 area-mipmap，即非负面积平均的多尺度图像金字塔。目标深度与源 RGB 保留梯度；层级、父掩码及软权重冻结。支持覆盖全部实际采样位置，外层始终除以六，空边为零。')
    p('极端 float32 投影溢出的 NaN 梯度问题已在正式准备前修复；旧源、成功收据和失败探针保留。两套同刻排程在各 3000 步块内精确保留原三列观察多重集合，三相机不同且原帧号相同，但不意味着每 100 步曝光相同。排程只使用标定与父模型粗重叠。')
    fd=native['target_xyz_FD']['relative_errors']
    p(f'原生高斯 CUDA 梯度 fixture 在 A100 GPU1 通过。目标 XYZ 两差分步长的相对误差为 {fd[0]:.5f}、{fd[1]:.5f}，源 RGB 为 {native["source_RGB_FD"]["relative_error"]:.5f}；辅助矩的协方差和不透明度梯度为空，主 RGB 路径正常求导。实际 2 RGB、6 矩前向、0 Adam，墙钟 {native["wall_seconds"]:.3f} 秒。它只证明有限窗口内的实现梯度。')
    p(f'1140 个合法训练观察的父 HR 原始矩已完整导出：1140 次原生矩前向、0 RGB、0 更新，总段墙钟 {parent["seconds"]:.3f} 秒，峰值 {parent["peak_gb"]:.3f} GB。A、M1、M2 是不透明度贡献总和及其加权深度一阶、二阶矩。先聚合原始矩再归一，可保留多个表面混合的深度分散程度；渲染矩无需读取训练 HR 图像。')
    cpucount=len(stop['closed_CPU_edge_records'])
    p(f'原 CPU 支持阶段已闭合 {cpucount} / 8570 条有向边，逐边 NPZ、JSON 与 SHA 保留，闭合边耗时合计 {stop["closed_CPU_edge_seconds_sum"]:.3f} 秒。根代理停止自己拥有的 CPU 准备服务，并在 A100 GPU1 以同公式恢复后续边；1140 导出及 fixture 均复用。被中断而未返回的 CPU 操作成本为未知。')
    p('执行设备改变没有改变深度分散尺度 τ、阈值、掩码或软权重公式。当前生成环境为 CUDA，缓存含 CPU 复用边；混合缓存耗时不能称为全 CUDA 成本，也没有宣称两设备结果逐位相同。未知方差保留中性权重 1，不作为明确遮挡证据。')
    p(f'远端完整准备已有精确退出码 0，完整支持及一次 train76 校准的元数据 SHA 已核验。train76 为 19 训练相机各 4 固定帧，只使用合法训练 LR/教师和冻结父模型。冻结 λX={cal["lambda_X"]:.8f}、κ={cal["kappa"]:.8f}、τ={cal["tau_z"]:.8f}，不使用 HR 或开发误差选权重。校准 {cal["cost"]["seconds"]:.3f} 秒、228 RGB、228 矩前向、0 Adam；父状态未改。')
    p('远端通过与本地可用分开验收。待回传整包为 15.45 GiB、21896 文件；本地完整 SHA 验收及准备 complete 发布仍未闭合。r1 后续和本地统一评价继续等待该完成事件，不能凭远端元数据跳过本地资产检查。')
    h('资源实际分配与预算分离')
    p('短窗 r1 六臂固定本机 PRO6000，r2 六臂固定 A100 GPU1，十二端点统一本机 RTX3090 评价。用户决定等待 A100 GPU0 的既有 openpi 服务退出后再使用，不共享、不抢占。每次派发重新核查占用，统一评价不能消除训练硬件引起的优化差异。')
    page()
    h('完整时间输入及独立模型前缀')
    table(['已用开发场景','相机流','全观察','训练观察'],[
        [s['scene'].replace('meetroom_','MeetRoom '),s['cameras'],s['all_observations'],s['train_observations']]
        for s in development['scenes']],[7.2,2.6,3.0,5.2])
    p(f'四场景完成 {development["total_cameras"]} 条相机流、{development["total_all_observations"]} 全时间观察，以及每场两 seed 共 {development["completed_sparse_initializations"]} 份合法 LR 稀疏初始化；训练观察共 {development["total_legal_train_observations"]}。点云可同值，独立模型仍须分别初始化模型、形变、Adam 与随机状态。只有 Cook seed20261007 开始原生独立前缀。')
    p('Cook seed20261007 已由原 coarse100 检查点实际续至 coarse3000，精确服务终态证明退出码 0。首次 CUDA 恢复曾因加载位置把随机状态字节张量转到 GPU 而失败；修复只将随机状态输入恢复为 CPU，源 SHA 前 12 位 ab5fd0313d37。兼容修订绑定旧 100 步检查点、旧源与失败记录，没有冷启动或改写原检查点。')
    if finepassed:
        p(f'fine14000 已有完整前缀 complete 回执及同一服务身份的精确退出码 0。作者 coarse3000 / fine14000、两阶段 Adam 重置及最后一次只反向不 Adam 的顺序保留。实计 17000 循环、34000 RGB、17000 反向和 16999 Adam，父状态 {prefixcomplete["topology"]["points"]} 点。全部四次尝试记录墙钟 {prefixcomplete["cost_all_attempts"]["seconds"]:.3f} 秒，包含失败；独立计费，不属于短窗 72000 更新。')
    else:
        p('fine14000 已登记原生 CUDA 续接，服务身份前 12 位 24c0807510e6；完整 17000 循环尚未验收。作者 coarse3000 / fine14000、阶段 Adam 重置、多视图增密统计及 fine 最后一次只反向而不 Adam 的顺序保留。完整预算为 34000 RGB、17000 反向和 16999 Adam，独立计费。精确退出验收服务已登记，以进程退出事件检查完整回执。')
    p('前缀只读合法训练 LR，不使用 HOI 或 HR 监督；官方 cam00 仅评价，cam01 在完整协议中合法训练。作者图像接口适配到项目 ×4 分辨率，并同步校正裁剪及像素中心内参；初始点来自合法 LR，属于披露的接口适配。完整原生 SR 续训还需新父模型对应的教师、支持及训练数据校准，旧短窗资产不能直接验收完整阶段。')
    teachers=reuse['copied_teacher_images']+t300['selected']+(t900batch['generated'] if t900batch else 0)
    p(f'Cook 完整教师精确复用 1140 旧图，并完成 cam01 首 300 新图。首 60 推理 {t60["original_generation_seconds"]:.3f} 秒，续接新增 240 张 {t300["original_generation_seconds"]:.3f} 秒，均在本机 RTX3090。'+(f'through900 已获精确退出码 0，新增 {t900batch["generated"]} 张、选集内复用 {t900batch["reused"]} 张，新增推理 {t900batch["original_generation_seconds"]:.3f} 秒。总库存 {teachers} / 6000 张，仍未完整；3090 后续留给统一评价。' if t900batch else f'through900 已登记运行。已闭合库存 {teachers} / 6000 张，未闭合输出不计完整缓存。'))
    page()
    h('完成事件与完整 SR 接口')
    p(f'r2 首次派发因日志目录不存在，在进入 Python 前失败；精确退出码为 1，0 GPU、0 正式更新，失败证据保留。目录补齐后同一 owned unit 恢复，当前 InvocationID {worker["invocation_id"]}，登记 PID {worker["main_pid"]}。该回执证明持久派发及身份绑定，不证明当前模型进度或完成。r1 仍等待本地准备资产。')
    p(f'统一本机 RTX3090 评价消费者已持久注册，PID {evalreg["pid"]}。它先等待本地 preparation/complete 文件事件，再按固定十二任务顺序评价，单卡不并行；退出和文件事件驱动，不以模型轮询发现完成。当前统一评价仍为 {endpoints} / 12，每小时检查仅作异常兜底。远端端点还需完整回传及同一环境评价。')
    p('完整原生 SR 的 full_refine、full_support_prepare 和 full_evaluate 接口已通过 CPU 合同检查。完整细化保留原生单 Adam 八参数组，继承模型、Adam 和随机状态；统一在 SR 边界仅清零三项 LR 密度统计，并按 SR 1–6000 重启作者增密/剪枝时钟，学习率继续 fine14000 后的游标。这是明确披露的新日程适配，不能称作者方法完全未改。')
    p('完整阶段须用 90115 点新父模型、完整合法教师、排程及候选冻结证据。需要 X 的候选另建自己的支持、τ 与训练校准，不能复用 U6000 短窗身份；缺依赖即拒绝训练。最多选两个完整候选，当前未选择或派发完整 SR 网格。完整 cam00 全 300 帧浮点 PSNR/SSIM/LPIPS、区域和频带评价接口已具备，但没有 GPU 完整 SR 或 300 帧质量结果。')
    p('完整细化的早期 CPU 测试隔离曾误触普通 CUDA 张量初始化；没有原生渲染、GPU Adam 或正式更新。物理卡及耗时未知，事故与失败成本已保存。后续通过的合同在隐藏 CUDA 的子进程内执行；CPU 通过不能替代原生 GPU 验收。')
    h('尚缺证据与结论边界')
    p('Coffee 与 Flame 原官方 Range 下载已失败，闭合块及部分数据保留，恢复工作继续，两未用确认场景均不称就绪。输入完成不等于完整教师、两独立模型或官方主基准完成。结论仍须结合 PSNR（像素保真）、SSIM（结构相似）、LPIPS（感知距离）及区域、时序、频带证据。当前不宣布 M/X 有效。')
    for key,path in [('generator',Path(__file__)),('layout_helper',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/report.py'),('prefix_source',ROOT/'experiments/dynamic_sr_multiview_footprint_20261007/full_prefix.py')]:
        raw=path.read_bytes();(frozen/(key+'.py')).write_bytes(raw)
        identities[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(raw))
    identities['reference']=dict(path=str(reference.relative_to(ROOT)),sha256=sha(reference.read_bytes()))
    manifest=dict(at_utc=at,identities=identities,scope='Actual local receipts; no GPU, SSH, publication or quality conclusion')
    dump(frozen/'manifest.json',manifest)
    facts=dict(formal_updates_lower_bound=formal,B0_accepted_updates=b0updates,training_endpoints=training_endpoints,
        uniform_evaluation_endpoints=endpoints,full_endpoints=endpoints,closed_CPU_edges=cpucount,parent_HR_moments=1140,
        full_prefix_complete_accepted=finepassed,remote_preparation_accepted=True,remote_calibration_passed=calpassed,
        local_preparation_accepted=localaccepted,local_return_expected_GiB=15.45,local_return_expected_files=21896,
        coefficients={k:cal[k] for k in ('lambda_X','kappa','tau_z')},closed_teacher_images=teachers,
        r2_dispatch_invocation=worker['invocation_id'],r2_dispatch_PID=worker['main_pid'],r2_dispatch_is_completion=False,
        full_refinement_CPU_contract_passed=True,full_evaluation_CPU_contract_passed=True,full_SR_GPU_quality_verified=False)
    dump(frozen/'content.json',dict(blocks=blocks,facts=facts))
    (frozen/'artifact.md').write_text(f'# Progress report edit contract\n\nReference {reference}\nSHA {identities["reference"]["sha256"]}\n\nOne Letter portrait section; margins side 1.8 cm and vertical 1.6 cm. Preserve styles, relationships, section geometry and page furniture. Noto Sans CJK SC body 11 pt / 15.5 pt leading, Title 23 pt, Heading 1 15 pt. All headings black, Title no border. Retained patterns: title and experiment table; operator/preparation narrative; complete-time input table and continuation narrative. Repeat the retained narrative pattern for the new completion/full-SR-interface page. Editable slots are body paragraphs/tables and core title/subject. Table component widths are deliberate, header repeats, borders D9D9D9, rows have no fixed height. Four-scene table reuses the retained table component. Reference package and frozen inputs must remain unchanged. Final pages require visual QA.\n')
    helper=module('fp_existing_document_layout',ROOT/'experiments/dynamic_sr_confidence_geometry_20261006/report.py')
    helper.save_markdown(STEM.with_suffix('.md'),blocks)
    temporary=frozen/'body_components.docx';helper.save_docx(temporary,blocks)
    from docx import Document
    from docx.oxml.ns import qn
    doc=Document(reference);generated=Document(temporary);body=doc.element.body
    for element in list(body):
        if element.tag!=qn('w:sectPr'):body.remove(element)
    for element in generated.element.body:
        if element.tag!=qn('w:sectPr'):body.insert(len(body)-1,copy.deepcopy(element))
    doc.core_properties.title='同刻多视图与像素足迹训练进度'
    doc.core_properties.subject='实际已接受检查点及准备进度 不作最终质量结论'
    doc.save(STEM.with_suffix('.docx'))
    for suffix in ('.docx','.md'):
        path=STEM.with_suffix(suffix);(frozen/path.name).write_bytes(path.read_bytes())
    receipt=dict(status='created_requires_all_page_visual_QA',at_utc=at,frozen_inputs=str(frozen.relative_to(ROOT)),
        inputs_manifest_sha256=sha((frozen/'manifest.json').read_bytes()),
        docx=dict(path=str(STEM.with_suffix('.docx').relative_to(ROOT)),sha256=sha(STEM.with_suffix('.docx').read_bytes())),
        markdown=dict(path=str(STEM.with_suffix('.md').relative_to(ROOT)),sha256=sha(STEM.with_suffix('.md').read_bytes())),
        source=identities['generator'],facts=facts)
    dump(frozen/'snapshot.json',receipt);dump(OUT/'progress_report_snapshot.json',receipt)
    for path in frozen.rglob('*'):
        if path.is_file():path.chmod(0o444)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
