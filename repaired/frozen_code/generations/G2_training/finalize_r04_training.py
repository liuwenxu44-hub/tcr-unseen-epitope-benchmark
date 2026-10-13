"""Close one explicit descriptive R04 release. This script never fits or scores a model."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,json,math,subprocess,sys

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def new(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def rows(p):
 with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def table(p,data,fields):
 with p.open('x',encoding='utf-8',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(data)
def f(x):return format(float(x),'.6f')
def signed(x):return format(float(x),'+.6f')

def main():
 w=Path(sys.argv[1]).resolve();tool=w/'authority/freeze_manifest.py'
 # Select only files named by producer-owned catalogs or explicit root paths.
 selected={};catalogs=['authority/authority_owned_catalog.json','additive/owned_catalog.json','metrics/metric_owned_catalog.json','execution/execution_output_catalog.json']
 def add(p,role,expected=None):
  q=w/p;assert q.is_file(),p
  entry={'path':p,'role':role,'bytes':q.stat().st_size,'sha256':sha(q)}
  if expected:assert entry['bytes']==expected['bytes'] and entry['sha256']==expected['sha256'],p
  if p in selected:assert selected[p]['sha256']==entry['sha256']
  else:selected[p]=entry
 for name in catalogs:
  for x in read(w/name)['files']:add(x['path'],x.get('role','OWNED_ANALYSIS_ARTIFACT'),x)
  add(name,'EXPLICIT_PRODUCER_CATALOG')
 pre=read(w/'pretraining_identity.json')
 assert pre['status']=='FROZEN_BEFORE_NUMERICAL_FITS'
 for x in pre['files']:add(x['path'],x['role'],x)
 ledger=read(w/'execution/fit_ledger.json');summary=read(w/'execution/execution_summary.json')
 tasks=read(w/'authority/task_mapping105.json')['tasks'];contract=read(w/'authority/model_contract.json')
 assert len(ledger)==len(tasks)==105 and {x['task_id'] for x in ledger}=={x['task_id'] for x in tasks}
 assert all(x['status']=='PASS' and x['fit_attempts']==1 and x['params']==contract['actual_estimator_params'] for x in ledger)
 assert summary['interaction_fit_calls']==105 and summary['failed_tasks']==0 and summary['additive_numerical_fits']==summary['CNN_fits']==summary['external_scores']==0
 assert summary['inferential_statistics_computed'] is False
 assert read(w/'execution/failure_records.json')==[]
 frozen_time=datetime.fromisoformat(read(w/'pretraining_manifest.json')['created_utc'])
 for t in tasks:
  attempt=read(w/('execution/tasks/'+t['recipe_id']+'/'+t['fold_id']+'/'+t['draw_id']+'/fit_attempt.json'))
  assert attempt['fit_invocations_registered']==1 and datetime.fromisoformat(attempt['utc_before_call'])>frozen_time
 meta=[read(w/x['metadata_file']) for x in ledger]
 assert all(x['feature_fit_partitions']==x['model_fit_partitions']==['train'] and x['sample_weight_passed'] is False and x['joint_additive_block_manually_modified'] is False for x in meta)
 assert all(x['converged'] and not x['warnings'] and max(x['iterations'])<5000 for x in meta)
 independent=read(w/'validation/model_roundtrip105.json');entities=read(w/'validation/additive_entities105.json');metrics=read(w/'validation/metric_validation_attempt1.json')
 static=read(w/'validation/no_fit_and_parent_static_gate.json');shared=read(w/'additive/shared_transform_validation.json')
 assert independent['status']==entities['status']==metrics['status']==static['status']=='PASS'
 assert independent['passing_models']==entities['passing_artifacts']==105 and independent['fit_calls']==entities['fit_calls']==0
 assert all(x['status']=='PASS' and x['logits_exact_equal'] and x['probabilities_exact_equal'] for x in independent['results'])
 assert metrics['failure_count']==0 and metrics['passing_checks']==metrics['check_count']
 assert static['original2041_exactly_unchanged'] and static['all105_loaded_interaction_predictions_bitwise_equal']
 assert shared['task_count']==105 and shared['status']=='PASS105_SHARED_TRAIN_ONLY_TRANSFORMS_AND_INDEPENDENT_ARTIFACTS'
 assert read(w/'authority/g2_manifest_after.json')['status']=='PASS' and read(w/'authority/g2_manifest_after.json')['files']==857
 assert read(w/'authority/protected_identity_after.json')['status']=='PASS'
 rr=rows(w/'metrics/recipe_summary.csv');dd=rows(w/'metrics/draw_oof_summary.csv');ss=rows(w/'metrics/vs_s0_recipe_summary.csv');sd=rows(w/'metrics/vs_s0_draw_summary.csv');coverage=rows(w/'metrics/coverage_audit.csv')
 assert len(rr)==len(ss)==7 and len(dd)==len(sd)==len(coverage)==21
 assert all(x['status']=='COMPLETE_DESCRIPTIVE' for x in rr+ss+dd+sd)
 assert all(x['observed_query_count']=='168' and x['observed_positive_count']=='10289' and x['balanced_query_count']=='168' and x['row_count']=='20578' and x['queries_in_multiple_test_folds']=='0' for x in coverage)
 assert len(rows(w/'metrics/per_query_oof.csv'))==len(rows(w/'metrics/vs_s0_per_query.csv'))==3528
 assert all(math.isfinite(float(x[k])) for x in rr for k in x if k.endswith(('_mean','_min','_max','_range','_sd_ddof0')))
 conv={'status':'PASS','tasks':105,'interaction_fit_calls':105,'failed_fits':0,'warnings':0,'iteration_min':min(min(x['iterations']) for x in meta),'iteration_max':max(max(x['iterations']) for x in meta),'joint_additive_block_norm_min':min(x['jointly_fitted_additive_block_l2_norm'] for x in meta),'joint_additive_block_norm_max':max(x['jointly_fitted_additive_block_l2_norm'] for x in meta),'joint_additive_block_nonzero_count':sum(x['jointly_fitted_additive_block_l2_norm']>0 for x in meta),'test_predictions_total':sum(x['test_rows'] for x in ledger),'independent_prediction_max_abs_delta':max(x['probability_max_abs_delta'] for x in independent['results']),'validation_fits':0}
 new(w/'decision/convergence_and_budget.json',conv)
 view=[]
 for x in ledger:
  view.append({k:x[k] for k in ['task_id','recipe_id','fold_id','draw_id','status','fit_attempts','model_seed','checkpoint_reuse','model_id','model_file','model_sha256','prediction_file','prediction_sha256','metadata_file','metadata_sha256','test_rows']}|{'iterations':';'.join(map(str,x['iterations'])),'warning_count':len(x['warnings']),'C':x['params']['C'],'solver':x['params']['solver'],'max_iter':x['params']['max_iter'],'tol':x['params']['tol']})
 table(w/'decision/fit_ledger105.csv',view,list(view[0]))
 operations={'status':'COMPLETE_NO_MODEL_FAILURES','interaction_failures':0,'additive_FAIL':0,'additive_UNVERIFIED':0,'metric_failures':0,'technical_events':[{'event':'ADDITIVE_PREPARE_PATH_INTERFACE','timing':'before admission/artifacts/performance','disposition':'prepare-only absolute-path mount replaced by fixed /g2 relative mapping; original code/protocol retained in additive/attempt1; no optimizer calls'},{'event':'PREDICTION_COLUMN_ORDER_ALIGNMENT','timing':'before pretraining freeze and first interaction fit','disposition':'producer column order probability,logit aligned to already-frozen metric contract; no numerical values existed'},{'event':'ADDITIVE_SSH_TRANSPORT_CLEANUP_255','timing':'after 105 analytic artifacts and shared-transform validation completed','disposition':'local residual SSH wrapper terminated after server computation ended. Wrapper exit255 is transport cleanup; do not claim its normal exit0. Artifact/hash/independent entity checks establish PASS.'}]}
 new(w/'decision/failures_and_operational_events.json',operations)
 # Printed values retain a full-precision, field-specific source crosswalk.
 claims=[]
 for name,data in [('metrics/recipe_summary.csv',rr),('metrics/vs_s0_recipe_summary.csv',ss),('metrics/draw_oof_summary.csv',dd),('metrics/vs_s0_draw_summary.csv',sd)]:
  for x in data:
   for k,v in x.items():
    if k in {'recipe_id','draw_id','status','fixed_draw_count','complete_draw_count','sd_ddof','query_count','row_count','npositive','nnegative','paired_query_count'}:continue
    claims.append({'claim_id':x['recipe_id']+':'+x.get('draw_id','three_draw_summary')+':'+k,'source_file':name,'source_sha256':sha(w/name),'selector':{z:x[z] for z in ('recipe_id','draw_id') if z in x},'field':k,'full_precision_value':v,'display_value':f(v),'display_policy':'six decimals, source precision unchanged','evidence':'DESCRIPTIVE_ONLY_NO_CI_NO_P_NO_TRUE_PAIRING_TEST'})
 new(w/'decision/claim_value_ledger.json',{'claims':claims,'acceptance_does_not_depend_on_metric_direction_or_significance':True})
 sby={x['recipe_id']:x for x in ss};dby={(x['recipe_id'],x['draw_id']):x for x in sd}
 lines=['# R04 G2 训练及描述性敏感性报告（2026-10-09）','',
 '本轮完成并停止：105/105 交互 Logistic 拟合及 105/105 独立解析 Additive 产物通过验收；七套方案、21 份 OOF 和全部逐 query 指标完整。该结果仅用于事后描述性敏感性分析。没有新增 P 值或推断性 CI，没有恢复 NOT_INFERENTIAL 检验，也没有据此选择新默认方案。','',
 '## 身份与范围','',
 '- 冻结 v1.1 SHA-256：`56b8d6d745528a6a7daa185186ab48faebf1f647296815104441c9db754b3713`。',
 '- 唯一数据入口 G2 manifest SHA-256：`98bfca307c68f01f6980f3f2f0d6bf01529254cd554ffff4a253d6fc54942d35`；857 文件训练前后均一致。',
 '- 任务映射：`authority/task_mapping105.json`；S0–S6 × 5 folds × 3 draws，每个任务 train/validation/test 使用同一 draw。原禁边规则及 UNKNOWN 的保守风险排除语义由 G2 继承，UNKNOWN 没有被标为阳性。本轮没有重做异常核查、重新分配或重新匹配。',
 '- 共同阳性 10,289 条、完整 query 168 个；每 recipe/draw 的 OOF 含 20,578 条（阳性/阴性各 10,289），每个 query 只出现在一个 test fold。21 份 OOF 共 432,138 条评分记录；3,528 条 query 指标及对应 S0 差值全部保留。',
 '- 训练方法冻结 `pretraining_manifest.json` 共 266 文件，生成早于全部 105 次 fit 登记。训练容器仅挂载 G2、当前源代码快照及本代输出；旧数据与模型未被训练器加载。','',
 '## 固定训练与解析对照','',
 '交互模型保持原 sklearn LogisticRegression：L2、C=1、liblinear、max_iter=5000、tol=0.0001、seed=13、单位权重、class_weight=None。S5/S6 的权重是分配目标权重，不是模型样本权重。HLA 映射仅由对应训练集建立；序列编码与 20×20 双中心化矩阵是固定变换。验证集仅使用训练映射作变换和审计，不选参、不提前停止。完整 16 项参数、源码快照、配置及 Docker image 身份见 `authority/model_contract.json`。','',
 '105 个新 Additive 训练集分别核验真实 TCR／完整 query 正负计数、有效权重、可分离特征、训练限定、L2 目标和 liblinear 截距惩罚及输入身份，全部 PASS。每任务独立构建解析零解，不调用数值 fit。所有 train/validation/test 实际特征评分严格 logit=0、概率=0.5，并通过保存重载和独立实体审查；准入不是根据预测接近 0.5 推断。交互模型共同拟合的 additive block 原样保留，105 个 block 范数均非零。','',
 '## 七套方案的主要结果','',
 '先在每个 draw 的完整 OOF 内按 query 等权计算 Macro，再对三个 draws 等权。下列括号为三个固定 draws 的最小值–最大值，全部是描述范围，不是置信区间。完整精度、ddof=0 标准差及 AUROC 对照差见 CSV。','',
 '| Recipe | Macro AP 均值（draw 范围） | Macro AUROC 均值（draw 范围） | ΔAP vs Additive | ΔAP vs 有限随机排序 | ΔAP vs G2 S0 | ΔAUROC vs G2 S0 |',
 '|---|---:|---:|---:|---:|---:|---:|']
 for x in rr:
  s=sby[x['recipe_id']]
  lines.append('| '+x['recipe_id']+' | '+f(x['interaction_AP_mean'])+' ('+f(x['interaction_AP_min'])+'–'+f(x['interaction_AP_max'])+') | '+f(x['interaction_AUROC_mean'])+' ('+f(x['interaction_AUROC_min'])+'–'+f(x['interaction_AUROC_max'])+') | '+signed(x['delta_Additive_AP_mean'])+' | '+signed(x['delta_finite_no_ties_AP_mean'])+' | '+signed(s['delta_G2_S0_interaction_AP_mean'])+' | '+signed(s['delta_G2_S0_interaction_AUROC_mean'])+' |')
 lines+=['',
 '所有 recipe/draw 的实际 query 都满足阳性数=阴性数，解析 Additive 的 Macro AP/AUROC 均为 0.5。有限无 ties 随机排序 AP 参考按每个 query 实际 P、N 分别计算：`H_n/n + (P-1)(n-H_n)/(n(n-1))`，n=P+N；再按相同 query/draw 权重汇总。因共同阳性和实际每 query 负例配额相同，七套方案的参考都为 '+f(rr[0]['finite_no_ties_expected_AP_mean'])+'，AUROC 参考为 0.5。这是均匀无 ties 排序的描述性期望，没有假设它是保留共享实体或真实配对信息的有效 null。','',
 'S1（阈值 0.4）的均值接近 G2 S0，S2（0.6）较低；不同真实分配 S3 较低、S4 略高；分配权重变化 S5/S6 较低。S3 的平均 AP 与有限排序参考几乎相同，S5 平均 AP 低于该参考。S5 三个 draws 的 AUROC 均低于 0.5，S3 的描述范围跨过 0.5。不能将所有方案均高于恒定 Additive 的 AP 写成一致支持真实配对信息，也不能据此宣称模型完全没有配对信号。','',
 '这些比较反映固定 recipe 整体造成的变化；分配变化伴随训练任务和负图变化，不能将差值拆解为单一参数的因果效应。S0 是本代 G2 S0，不能冒充原冻结内部队列的 R02 结果。三个 draws 共用阳性和部分结构，不是三个独立生物学重复；SD 和范围也不涵盖来源依赖、重新抽取队列或总体重训不确定性。原 R03 的来源局限及原检验的 NOT_INFERENTIAL 身份保持有效。','',
 '## 每个 draw 的结果','',
 '| Recipe | Draw | Macro AP | Macro AUROC | ΔAP vs Additive | ΔAP vs 有限排序 | ΔAP vs G2 S0 | ΔAUROC vs G2 S0 |',
 '|---|---|---:|---:|---:|---:|---:|---:|']
 for x in dd:
  s=dby[(x['recipe_id'],x['draw_id'])]
  lines.append('| '+' | '.join([x['recipe_id'],x['draw_id'],f(x['interaction_AP']),f(x['interaction_AUROC']),signed(x['delta_Additive_AP']),signed(x['delta_finite_no_ties_AP']),signed(s['delta_G2_S0_interaction_AP']),signed(s['delta_G2_S0_interaction_AUROC'])])+' |')
 lines+=['','## 验收与失败登记','',
 f'- 拟合账本：105 次交互 fit、105 PASS、0 FAIL，迭代数 {conv["iteration_min"]}–{conv["iteration_max"]}，0 警告；Additive fit、CNN fit、外部逐臂评分均为 0。没有重新拟合或失败任务替换。',
 '- 独立交互模型检查：105 实体、4,325 项检查全部 PASS；重新建立训练 HLA 映射、重建 G2 test 特征，保存的全部 logit/probability 逐值精确一致。',
 '- 独立 Additive 实体检查：105 实体、4,955 项检查全部 PASS；105 共享训练映射／输入／配置核验全部 PASS；协议的 10 项拒绝准入回归、有限 AP 数学自检及缺失任务传播 fixture 通过。',
 '- 指标独立实现：13,362 项检查全部 PASS；以完整分数组处理 ties，独立手工 AP/AUROC 从源测试记录重算，绝对容差 1e-12。',
 '- 原父代 2,041 文件逐项未变；after 保护联合清单 2,071 文件全部 PASS，增加的 30 项是本轮代码／快照引用。未覆盖旧模型、预测、图表或正文。',
 '- 计算阶段指标 summary 的独立验证字段保留当时 PENDING；后续 `validation/metric_validation_attempt1.json`、两个实体复核报告和本轮最终决定共同闭合验收，未改写阶段记录。',
 '- 技术记录单独保留：Additive 的 prepare-only 路径接口修订、训练前 CSV 列序对齐；均发生在相关结果之前。Additive 计算完成后的残留本地 SSH 清理返回 255，未捕获该 wrapper 的正常退出码，不能声称该传输 wrapper exit0；105 实体及其哈希／独立验收均已通过。详见 `decision/failures_and_operational_events.json`。',
 '- 因本轮拟合总上限为 105，没有另起两套模型重训来验证可复现性；采用冻结代码／环境、所有模型序列化重载、独立特征评分及两种指标实现核验，验证阶段新增 fit 为 0。','',
 '## 固定交付与边界','',
 '`execution/fit_ledger.json` 和 `decision/fit_ledger105.csv` 为逐任务账本；`execution/tasks/S*/fold_*/draw_*/` 保存模型、原始预测、metadata 和 fit 登记。`additive/` 保存独立解析模型、逐节点计数和九门准入；`metrics/oof/` 保存 21 份两模型配对 OOF。完整逐 query、draw、recipe、实际有限 AP 参考、对照差、S0 同 draw 差及覆盖检查均在 `metrics/`。',
 '',
 '最终证据入口为 `decision/final_release_decision.json` 与 `r04_training_manifest.json`，后者只冻结显式清单选择的文件。数值与来源字段对应见 `decision/claim_value_ledger.json`。本轮验收仅依赖身份、覆盖、算法及预算规则，不依赖显著性、效应方向或能否保留原结论。没有修改稿件图表，没有启动 CNN、外部逐臂评分或后续返修阶段。']
 with (w/'decision/R04_G2_SENSITIVITY_REPORT.md').open('x',encoding='utf-8') as out:out.write('\n'.join(lines)+'\n')
 decision={'status':'PASS_DESCRIPTIVE_R04_G2_ONLY','utc':datetime.now(timezone.utc).isoformat(),'required_tasks':105,'interaction_fit_calls':105,'successful_tasks':105,'failed_tasks':0,'Additive_numerical_fits':0,'OOF_count':21,'queries_per_OOF':168,'query_metric_rows':3528,'independent_model_checks':independent['actual_check_count'],'independent_Additive_checks':entities['actual_check_count'],'independent_metric_checks':metrics['check_count'],'g2_files_unchanged':857,'original_parent_files_unchanged':2041,'training_methods_pre_frozen':True,'fit_attempts_after_method_freeze':True,'analytic_admission_all_PASS':True,'all_joint_additive_blocks_preserved':True,'validation_fits':0,'CI_or_P_added':False,'external_or_CNN_added':False,'manuscript_or_figure_changed':False,'performance_selected_default':False,'acceptance_does_not_depend_on_metric_direction_or_significance':True,'closure_reports':['validation/model_roundtrip105.json','validation/additive_entities105.json','validation/metric_validation_attempt1.json','validation/no_fit_and_parent_static_gate.json','additive/shared_transform_validation.json'],'stage_pending_resolved_by_later_reports_without_overwrite':True,'scope_limitations':['Posthoc descriptive fixed-grid only, no population CI/P or true-pairing null.','Three fixed draws are not independent biological replicates.','Common G2 analysis universe differs from original R02 internal frozen universe.','No second training rebuild:105 fit budget exhausted exactly; independent reconstruction and serialization/metric checks substituted within authorized scope.'],'next_action':'STOP_THIS_ROUND; retain artifacts for later separately authorized manuscript work.'}
 new(w/'decision/final_release_decision.json',decision)
 rootfiles=['run_g2_training.py','launch_g2_training.sh','training_protocol.json','freeze_pretraining.py','finalize_r04_training.py','pretraining_identity.json','pretraining_manifest.json','training_stdout.log','authority/freeze_manifest.py','validation/pretraining_manifest_check.json','decision/convergence_and_budget.json','decision/fit_ledger105.csv','decision/failures_and_operational_events.json','decision/claim_value_ledger.json','decision/R04_G2_SENSITIVITY_REPORT.md','decision/final_release_decision.json']
 for p in rootfiles:add(p,'ROOT_RELEASE_METHOD_REPORT_OR_EVIDENCE')
 # Recursive listing is an inventory-only guard, never a way of selecting inputs.
 unexpected=[];excluded=[]
 for p in w.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(w).as_posix()
  if rel in selected:continue
  if '__pycache__/' in rel or rel.startswith('scratch/') or rel=='authority/README_source_snapshot.txt':excluded.append(rel)
  else:unexpected.append(rel)
 assert not unexpected,('Unexpected files: inspect before freeze',unexpected)
 inv={'status':'PASS_CLOSED_EXPLICIT_INVENTORY','selected_file_count':len(selected),'excluded_runtime_cache_or_scratch':excluded,'unexpected_files':unexpected,'selection':'Producer catalogs plus explicit root files. Directory walk used only to detect unexpected user files.'}
 new(w/'decision/closed_inventory.json',inv);add('decision/closed_inventory.json','INVENTORY_GATE')
 new(w/'decision/release_owned_catalog.json',{'status':'CLOSED_EXPLICIT_RELEASE','files':list(selected.values()),'self_excluded':True})
 add('decision/release_owned_catalog.json','RELEASE_FILE_CATALOG')
 args=[sys.executable,str(tool),'create','--root',str(w),'--output','r04_training_manifest.json','--analysis-id','R04_G2_TRAINING_DESCRIPTIVE_SENSITIVITY_20261009','--status','frozen','--allow-dirty','--note','Final descriptive-only release after all105 model/analytic and independent metric checks. No CI/P/extra training. Parent files unchanged. Selection from explicit producer catalogs; existing local Git dirtiness preserved.']
 for x in selected.values():args+=['--file',x['role']+'='+x['path']]
 subprocess.run(args,check=True)
 subprocess.run([sys.executable,str(tool),'check','--root',str(w),'--manifest',str(w/'r04_training_manifest.json'),'--report',str(w/'validation/final_manifest_server_check.json')],check=True,stdout=subprocess.DEVNULL)
 print(json.dumps({'status':decision['status'],'files':len(selected),'manifest_sha256':sha(w/'r04_training_manifest.json')}))
if __name__=='__main__':main()
