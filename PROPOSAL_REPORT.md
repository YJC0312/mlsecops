# 糖尿病篩檢模型提案報告

> 對應 SDG：**SDG 3 — Good Health and Well-being（健康與福祉）**
>
> 依據 `notebooks/01_final_screening_analysis.ipynb` 與 `notebooks/01_final_screening_analysis_zh.ipynb` 的最終分析結果撰寫。

---

## 摘要

本研究以 CDC BRFSS 2015 糖尿病健康指標資料集建立糖尿病二元初篩模型。研究重點不是單純追求最高 Recall 或最高 Accuracy，而是回到「初篩」這個使用情境：模型必須盡量抓出潛在糖尿病個案，同時避免假陽性過多，否則會造成不必要的轉介、檢驗與醫療資源負擔。

特徵診斷再決定哪些欄位適合前處理；模型比較則使用同一套 train / validation / test 切分；threshold 選擇也不直接採用預設 `0.5`，而是在 validation set 上以 `Recall >= 0.75` 作為最低條件，再挑選 Precision 最高的 operating point。

最終選定模型為 **XGBoost 搭配 `scale_pos_weight` 類別不平衡策略**。在 validation set 上，模型以 threshold `0.542078` 達成 Recall `0.7501`、Precision `0.3244`、Specificity `0.7471`、ROC-AUC `0.8283`。將模型與 threshold 凍結後，在 held-out test set 上得到 Recall `0.7512`、Precision `0.3244`、Specificity `0.7467`、F1 `0.4531`、F2 `0.5947`、ROC-AUC `0.8267`、PR-AUC `0.4229`。混淆矩陣為 TP `5,310`、FP `11,061`、TN `32,606`、FN `1,759`。

這些結果表示模型能抓出約 75% 的陽性個案，同時保留約 75% 的陰性辨識能力。不過 Precision 約 32%，代表模型預測為高風險者中仍有相當比例是假陽性。因此本模型最適合定位為**第一層風險篩檢與轉介輔助工具**，而不是診斷系統。

---

## 1. 研究動機與問題定義

糖尿病是全球重要慢性病之一，若未及早發現與介入，可能導致心血管疾病、腎臟病變、視網膜病變、神經病變等長期併發症。傳統診斷仰賴 HbA1c、空腹血糖或口服葡萄糖耐量測試，這些檢驗具有臨床可靠性，但需要醫療資源、抽血或特定流程，不一定適合大規模社區初篩。

本研究的問題設定是：是否能利用非侵入式問卷資料與基本健康指標，建立一個可重現的機器學習模型，用於糖尿病風險初步分流？這個問題的重點不是取代醫師或實驗室檢驗，而是在資源有限的情境中，協助找出較需要後續檢查的人。

這個任務有三個關鍵挑戰。第一，資料高度不平衡，陽性比例約 13.93%，因此 Accuracy 容易被多數類別主導。第二，初篩場景重視 Recall，因為漏掉高風險個案會延誤後續檢查；但 Precision 也不能太低，否則大量假陽性會降低系統實用性。第三，模型選擇不能只看某一個單點指標，而要明確說明 threshold、混淆矩陣與臨床使用代價。

基於上述原因，本研究採用 Precision-Recall 取向，而不是 Accuracy 取向。具體做法是先設定最低 Recall 條件，再在符合條件的候選模型中選擇 Precision 較高者。這樣的設計較符合初篩工具的真實決策邏輯。

---

## 2. 研究目標與決策原則

本研究目標不只是訓練一個模型，而是建立一套從資料診斷到最終 test evaluation 都可追溯的篩檢流程。每個步驟都必須回答三個問題：為什麼這樣做、為什麼這樣選、結果如何。

具體目標如下：

1. 建立防止資料洩漏的糖尿病二元分類流程。
2. 在建模前驗證 schema、缺失值、重複列、類別比例與連續欄位極端值。
3. 依照欄位型態決定前處理方式，避免把問卷中的二元或有序欄位當成一般連續變數任意裁切。
4. 比較多種不平衡處理策略，包括 `class_weight`、`scale_pos_weight`、SMOTE pipeline 與 SMOTENC pipeline。
5. 使用 validation set 選擇 threshold，規則為「在 Recall >= 0.75 的候選中選 Precision 最高者」。
6. 將模型與 threshold 凍結後，只在 held-out test set 上做一次最終評估。
7. 清楚說明模型適用於初篩與轉介輔助，不適合作為最終診斷。

這些原則讓分析流程具有可追溯性：資料健康度決定切分與不平衡策略，特徵診斷決定前處理候選，cross-validation 與 validation ranking 決定模型與 threshold，最後才在 test set 做一次確認。

---

## 3. 資料集與資料健康度

本研究使用 Kaggle 公開資料集 **CDC Diabetes Health Indicators Dataset**，原始來源為美國 CDC BRFSS 2015 問卷調查。資料包含自評健康、慢性病史、生活習慣、BMI、年齡、教育與收入等欄位，適合用於非侵入式風險篩檢研究。

| 項目 | 數值 |
| --- | ---: |
| 樣本數 | 253,680 |
| 欄位數 | 22 |
| 特徵數 | 21 |
| 目標欄位 | `Diabetes_binary` |
| 缺失值 | 0 |
| 重複列 | 24,206 |
| 陽性比例 | 13.93% |

目標欄位 `Diabetes_binary` 定義如下：

| 類別 | 意義 |
| --- | --- |
| 0 | 無糖尿病 |
| 1 | 糖尿病前期或糖尿病 |

資料健康度診斷直接影響後續設計。首先，缺失值為 0，因此不需要額外設計 imputation 流程，減少一層可能引入偏差的處理。其次，陽性比例只有 13.93%，因此 Accuracy 不能作為主指標，否則模型即使偏向預測陰性也可能看起來表現不差。第三，重複列有 24,206 筆，但 BRFSS 是問卷資料，不同受訪者可能具有相同回答組合；直接刪除可能改變族群分布，因此本研究先保留重複列，並把去重視為後續敏感度分析。

這一步的結論是：資料本身足以進行建模，但模型評估必須以 Precision、Recall、PR-AUC、Specificity 與混淆矩陣為主，不能只看 Accuracy 或 ROC-AUC。

---

## 4. EDA 與前處理決策

本資料集的欄位並非同質的連續變數。多數欄位是 0/1 二元問卷答案，例如是否高血壓、是否高膽固醇、是否運動；部分欄位是有序問卷等級，例如 `GenHlth`、`Age`、`Education`、`Income`；真正適合檢查極端值的欄位主要是 `BMI`、`MentHlth`、`PhysHlth`。

Notebook 診斷結果如下：

| Feature | Min | Q1 | Median | Q3 | Q99 | Max | 判讀 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BMI` | 12 | 24 | 27 | 31 | 50 | 98 | 明顯右尾，極端值需留意 |
| `MentHlth` | 0 | 0 | 0 | 2 | 30 | 30 | 問卷上限 30 天 |
| `PhysHlth` | 0 | 0 | 0 | 3 | 30 | 30 | 問卷上限 30 天 |

這些結果導出三個前處理選擇。第一，二元欄位與有序欄位只做 schema 與值域驗證，不能用一般 outlier clipping 處理，因為 `Age=13` 或 `GenHlth=5` 是合法問卷值，不是異常值。第二，`MentHlth` 與 `PhysHlth` 的最大值 30 是問卷設計上限，不能因為數值大就直接裁切。第三，`BMI` 的最大值 98 明顯高於 Q99 的 50，後續若要測試 capping，應只針對 `BMI`，且 cap 必須只從 train set 估計，不能用 validation 或 test 的分位數。

最終模型是樹模型 XGBoost，因此不需要對所有特徵做 standardization。這樣做的理由是樹模型依據切分點建立規則，對單調縮放不敏感；保留原始欄位反而更容易對應問卷語意。線性模型如 Logistic Regression 則可在其 pipeline 內做 scaling，避免不同模型需要的前處理互相污染。

---

## 5. 資料切分與防洩漏設計

本研究使用 stratified train / validation / test 切分，並維持三份資料的陽性比例一致。切分結果如下：

| Split | X shape | 陽性比例 |
| --- | --- | ---: |
| Train | `(162355, 21)` | 13.9337% |
| Validation | `(40589, 21)` | 13.9323% |
| Test | `(50736, 21)` | 13.9329% |

這樣切分的原因是，模型訓練、threshold 選擇與最終評估應該扮演不同角色。Train set 用於模型 fitting 與 cross-validation；validation set 用於比較候選模型、選擇 threshold 與決定最終 operating point；test set 必須在模型與 threshold 都固定後才使用，否則 test 結果會因為反覆調整而變得過度樂觀。

防資料洩漏是本流程的重要設計。SMOTE、scaling、capping 等任何需要從資料估計參數的步驟，都不能在切分前對整份資料執行。例如若先對整份資料做 SMOTE，再切分 train/test，合成樣本可能與 test 中原始樣本過度相似，導致模型表現被高估。因此本研究將不平衡處理放在訓練流程或 cross-validation folds 內，test set 永遠保持原始分布。

結果上，三份資料的陽性比例都約 13.93%，表示 stratification 成功保留原始類別分布。這讓 validation threshold selection 與 final test evaluation 更接近真實部署情境。

---

## 6. 候選模型與不平衡策略

實驗比較六個候選組合：

| Model | Strategy | 選入原因 |
| --- | --- | --- |
| Logistic Regression | `class_weight=balanced` | 線性基準模型，可檢查資料是否已有可分訊號 |
| Random Forest | `class_weight=balanced_subsample` | 非線性樹模型基準，可處理特徵交互作用 |
| XGBoost | `scale_pos_weight` | 強表格資料模型，能直接處理類別不平衡權重 |
| LightGBM | `scale_pos_weight` | 另一個梯度提升樹模型，用來檢查 XGBoost 結果是否穩定 |
| XGBoost | SMOTE in Pipeline | 測試合成少數類樣本是否改善篩檢表現 |
| LightGBM | SMOTENC in Pipeline | 對類別欄位較友善的 SMOTE 變體 |

選擇這些模型的原因，是要建立從簡單到複雜的效能階梯。Logistic Regression 提供最低複雜度基準；Random Forest 檢查非線性是否重要；XGBoost 與 LightGBM 代表表格資料中常見的高效梯度提升方法；SMOTE 與 SMOTENC 則檢查資料層級 oversampling 是否比模型層級 weighting 更有效。

GA-XGBoost 與 Stacking Ensemble 也可以作為進階延伸方法，但它們應在基礎模型已建立清楚效益後再加入。如果單一 XGBoost 已在 validation ranking 中表現最佳，直接加入 GA 或 Stacking 可能增加計算成本與解釋難度，卻未必改善 Precision-Recall trade-off。因此，本研究先採用較簡潔、可重現且表現穩定的單一 XGBoost 模型作為主方案。

---

## 7. Cross-Validation 結果與解讀

Cross-validation 指標如下：

| Model | Strategy | ROC-AUC | PR-AUC | Recall | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | `scale_pos_weight` | 0.8308 | 0.4353 | 0.7960 | 0.3071 | 0.4432 |
| XGBoost | `scale_pos_weight` | 0.8314 | 0.4352 | 0.7955 | 0.3080 | 0.4440 |
| Random Forest | `class_weight=balanced_subsample` | 0.8275 | 0.4270 | 0.7079 | 0.3413 | 0.4605 |
| XGBoost | SMOTE in Pipeline | 0.8234 | 0.4158 | 0.3077 | 0.4826 | 0.3758 |
| LightGBM | SMOTENC in Pipeline | 0.8186 | 0.4124 | 0.5111 | 0.3974 | 0.4470 |
| Logistic Regression | `class_weight=balanced` | 0.8239 | 0.4065 | 0.7660 | 0.3127 | 0.4441 |

這張表有幾個重要訊息。第一，XGBoost 與 LightGBM 的 ROC-AUC 都約 0.83，PR-AUC 都約 0.435，代表梯度提升樹確實適合這份表格資料。第二，Logistic Regression 的 Recall 也能達到 0.7660，表示資料中的主要風險因子具有明顯線性訊號；但 PR-AUC 較低，代表它在排序陽性個案方面不如 boosting 模型。第三，SMOTE 類策略雖然 Precision 較高，但 Recall 明顯不足，特別是 XGBoost + SMOTE 的 CV Recall 只有 0.3077，不符合初篩任務。

因此，cross-validation 的結論不是「Precision 最高就是最好」，而是「必須先符合初篩 Recall 需求」。在這個條件下，`scale_pos_weight` 類策略比 SMOTE 類策略更符合本研究目的。它不需要生成合成樣本，部署上也較簡單，且在 Recall 與 PR-AUC 上維持穩定。

---

## 8. Validation Threshold Selection 與模型選擇

模型輸出的機率不是最終分類。若直接使用預設 threshold `0.5`，不一定符合初篩場景。threshold 太低會提高 Recall 但增加假陽性；threshold 太高會提高 Precision 但漏掉更多陽性個案。因此，本研究使用 validation set 進行 operating point selection。

threshold selection rule 為：

> 在 validation set 上找出所有 Recall >= 0.75 的 threshold，並選擇其中 Precision 最高者。

Validation 排名如下：

| Rank | Model | Strategy | Threshold | Accuracy | Precision | Recall | Specificity | F1 | F2 | ROC-AUC | PR-AUC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | XGBoost | `scale_pos_weight` | 0.5421 | 0.7475 | 0.3244 | 0.7501 | 0.7471 | 0.4529 | 0.5942 | 0.8283 | 0.4358 |
| 2 | LightGBM | `scale_pos_weight` | 0.5424 | 0.7472 | 0.3241 | 0.7503 | 0.7467 | 0.4527 | 0.5941 | 0.8279 | 0.4371 |
| 3 | Random Forest | `class_weight=balanced_subsample` | 0.4593 | 0.7422 | 0.3192 | 0.7505 | 0.7409 | 0.4479 | 0.5908 | 0.8252 | 0.4293 |
| 4 | Logistic Regression | `class_weight=balanced` | 0.5114 | 0.7382 | 0.3153 | 0.7505 | 0.7362 | 0.4440 | 0.5881 | 0.8212 | 0.4032 |
| 5 | XGBoost | SMOTE in Pipeline | 0.2311 | 0.7349 | 0.3121 | 0.7501 | 0.7324 | 0.4408 | 0.5857 | 0.8203 | 0.4128 |
| 6 | LightGBM | SMOTENC in Pipeline | 0.3231 | 0.7343 | 0.3116 | 0.7501 | 0.7317 | 0.4403 | 0.5853 | 0.8172 | 0.4105 |

最終選擇 **XGBoost + `scale_pos_weight`**，threshold 固定為 `0.5420781373977661`。選擇它的原因有三點。第一，它在所有滿足 Recall 約束的候選中 Precision 最高。第二，它的 Specificity 也維持在 0.7471，代表沒有為了 Recall 完全犧牲陰性辨識能力。第三，它與 LightGBM 表現非常接近，但 XGBoost 在 Precision、F1、F2、ROC-AUC 上略高，且差距雖小但方向一致。

這個結果也說明 threshold selection 的價值：模型不是只輸出一個固定答案，而是可以依照篩檢目標調整 operating point。本研究選擇 Recall 0.75 作為平衡點，讓模型保留足夠敏感度，同時把假陽性控制在較可解釋的範圍內。

---

## 9. Held-Out Test 最終評估

模型與 threshold 凍結後，在 test set 上得到以下結果：

| 指標 | 數值 |
| --- | ---: |
| Model | XGBoost |
| Strategy | `scale_pos_weight` |
| Threshold | 0.542078 |
| Accuracy | 0.7473 |
| Precision | 0.3244 |
| Recall / Sensitivity | 0.7512 |
| Specificity | 0.7467 |
| F1 | 0.4531 |
| F2 | 0.5947 |
| ROC-AUC | 0.8267 |
| PR-AUC | 0.4229 |
| TP | 5,310 |
| FP | 11,061 |
| TN | 32,606 |
| FN | 1,759 |

test 結果與 validation 結果相當接近，這是重要訊號。Validation Recall 為 0.7501，test Recall 為 0.7512；Validation Precision 為 0.3244，test Precision 也為 0.3244；Validation Specificity 為 0.7471，test Specificity 為 0.7467。這表示所選 threshold 沒有明顯 overfit validation set，模型在未參與選擇的 test set 上維持穩定。

混淆矩陣的實務解讀如下：

- TP = 5,310：模型成功找出 5,310 位糖尿病或糖尿病前期個案，這些人可被建議進一步檢驗。
- FN = 1,759：仍有 1,759 位陽性個案未被抓出，代表模型不能單獨作為排除疾病的工具。
- FP = 11,061：假陽性數量不低，因此模型輸出應該是「建議追蹤」而不是「判定罹病」。
- TN = 32,606：模型能正確排除多數陰性個案，Specificity 約 0.7467。

F2 為 0.5947，高於 F1 的 0.4531，代表在重視 Recall 的評估下，模型表現較符合初篩目標。不過 PR-AUC 為 0.4229，也提醒我們：在低盛行率資料中，要同時取得高 Recall 與高 Precision 很困難，後續若要部署，應搭配二階段流程，例如先用模型篩出高風險者，再做 HbA1c 或空腹血糖確認。

---

## 10. 最終方案與臨床定位

最終方案為：

```text
Validated BRFSS 2015 Data
        -> Data Health and Feature Diagnostics
        -> Stratified Train / Validation / Test Split
        -> Leakage-Safe Imbalance Strategy Comparison
        -> XGBoost with scale_pos_weight
        -> Validation Threshold Selection: highest Precision under Recall >= 0.75
        -> Frozen Threshold Test Evaluation
```

這個流程的設計重點是可追溯性。資料診斷說明為什麼不能看 Accuracy；欄位診斷說明為什麼不對所有特徵做 clipping；cross-validation 說明為什麼 weighting 比 SMOTE 更符合目標；validation threshold selection 說明為什麼 threshold 不是任意決定；test evaluation 則確認選好的規則在未見資料上仍穩定。

臨床與公衛定位如下：

- 適合用於社區健康問卷、健檢前風險提醒、基層醫療轉介建議。
- 適合做第一層篩檢，協助決定誰應優先接受血糖或 HbA1c 檢查。
- 不適合單獨作為診斷依據，也不應直接決定治療。
- 若部署於非美國族群或不同年份資料，必須重新驗證，因為 BRFSS 2015 反映的是特定資料來源與族群結構。

---

## 11. 模型複雜度與延伸方法定位

本研究優先採用能清楚解釋、容易重現且驗證結果穩定的模型。GA-XGBoost、Stacking Ensemble 與 SHAP 都具有研究價值，但它們在流程中的角色不同。

| 方法 | 在本研究中的定位 | 理由 |
| --- | --- | --- |
| XGBoost + `scale_pos_weight` | 主模型 | 在 validation ranking 中達成最佳 Precision-Recall 平衡，且 test 結果穩定 |
| GA-XGBoost | 後續超參數搜尋方法 | 可在需要進一步提升 XGBoost 時加入，但會增加計算成本 |
| Stacking Ensemble | 後續模型融合方法 | 只有在不同基模型錯誤型態互補時才有明確價值 |
| SHAP | 後續解釋工具 | 應針對 frozen final model 進行解釋，讓 feature importance 對應真正採用的篩檢規則 |

這樣的安排讓模型複雜度服務於研究目標，而不是為了使用複雜方法而增加流程負擔。若未來要加入 GA 或 Stacking，應以 validation Precision、Recall、F2、Specificity 與 test 穩定性作為是否採用的判準。

---

## 12. 限制與後續工作

模型仍有明確限制。第一，Precision 約 32%，代表若模型標記 100 位高風險者，約 32 位是真陽性，仍有大量假陽性需要後續檢驗排除。第二，Recall 約 75%，代表仍有約四分之一陽性個案可能被漏掉，若實務場景更重視敏感度，可能需要測試 Recall 0.80 的 threshold，但必須接受 Precision 下降。第三，資料來自 BRFSS 2015，美國問卷族群不一定能直接代表台灣或其他地區。

後續工作如下：

1. **SHAP 解釋**：只針對 frozen XGBoost model 與 held-out examples 產生 feature importance 與個案解釋，檢查是否符合已知臨床風險因子。
2. **Threshold sensitivity analysis**：比較 Recall 0.70、0.75、0.80 對 Precision、FP、FN 的影響，提供不同部署情境的 operating point。
3. **Calibration check**：檢查預測機率是否需要 Platt scaling 或 isotonic calibration，讓風險分數更容易解讀。
4. **Subgroup analysis**：依年齡、性別、BMI 區間檢查模型是否對特定族群表現較差。
5. **External validation**：使用其他年份或其他族群資料驗證泛化能力。
6. **Two-stage screening design**：把本模型作為第一階段，第二階段接 HbA1c 或空腹血糖檢查，評估整體成本與效益。

---

## 13. 結論

本研究完成一套以 Precision-Recall 為核心的糖尿病初篩模型流程。最終結果顯示，**XGBoost + `scale_pos_weight`** 在 validation set 中取得最佳 threshold ranking，並在 held-out test set 上維持穩定表現：Recall `0.7512`、Precision `0.3244`、Specificity `0.7467`、ROC-AUC `0.8267`。

本研究的主要貢獻不只是選出一個模型，而是建立清楚的決策鏈：資料診斷說明評估指標選擇，特徵型態說明前處理策略，cross-validation 說明候選模型表現，validation threshold selection 說明 operating point，test evaluation 則確認最終結果。這使得糖尿病預測從單純模型比較，推進到更接近真實公衛篩檢部署的決策流程。

整體而言，本模型適合作為第一層風險篩檢工具。它能以非侵入式資料找出多數潛在陽性個案，但仍需要後續醫療檢驗與專業判讀。若未來能加入 SHAP 解釋、threshold sensitivity、校準與外部驗證，將更能支撐實際部署與臨床溝通。
