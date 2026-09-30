# src/ml/risk_tool.py
# 导入依赖库
import numpy as np
from xgboost import XGBClassifier
import structlog
from pathlib import Path

from src.config import settings

logger = structlog.get_logger(__name__)

# 训练好的XGBoost模型文件路径（risk_modeling notebook 训练产出，路径走 config 的 RISK_MODEL_PATH）
MODEL_PATH = Path(settings.risk_model_path)

# 模型要求输入的15个特征列表，顺序必须和训练时完全保持一致。
# 训练(risk_modeling notebook)与推理(本模块)分离，两边各自维护此列表，需人工保持一致。
FEATURE_COLS = [
    "age", "gender_m", "admission_type_emergency",
    "los_days", "diagnosis_count",
    "has_diabetes", "has_heart_failure", "has_hypertension",
    "has_renal_disease", "has_pneumonia", "has_sepsis", "has_copd",
    "num_icu_stays", "total_icu_los", "max_icu_los",
]


class ReadmissionRiskTool:
    """
    将训练完成的XGBoost 30‑天再入院预测模型封装成可调用工具
    供推理Agent智能体调用：当用户提问中包含病人信息时，Agent可调用本工具计算再入院风险
    """

    def __init__(self):
        # 模型对象初始化为空
        self.model = None
        # 实例化对象时自动加载本地模型文件
        self._load_model()

    def _load_model(self):
        """私有方法：从本地json文件加载训练好的XGBoost模型"""
        # 判断模型文件是否存在
        if not MODEL_PATH.exists():
            logger.warning("risk_model_not_found", path=str(MODEL_PATH))
            return
        # 创建空白XGBoost分类器实例
        self.model = XGBClassifier()
        # 从磁盘加载模型权重
        self.model.load_model(str(MODEL_PATH))
        logger.info("risk_model_loaded")

    def is_available(self) -> bool:
        """检查模型是否加载成功，返回布尔值"""
        return self.model is not None

    def predict(self, patient_profile: dict) -> dict:
        """
        根据病人信息字典，预测患者30天再入院风险

        :param patient_profile: 病人信息字典；字段全部为可选，缺失字段会自动填充默认值
            可用键名：age, gender_m, admission_type_emergency, los_days,
            diagnosis_count, has_diabetes, has_heart_failure,
            has_hypertension, has_renal_disease, has_pneumonia,
            has_sepsis, has_copd, num_icu_stays, total_icu_los, max_icu_los

        :return: 结果字典：包含风险分数、风险等级、临床解读文本
        """
        # 如果模型加载失败，直接返回错误信息
        if not self.is_available():
            return {
                "error": "Risk model not available. Run train_risk_model.py first.",
                "risk_score": None,
                "risk_level": None,
            }

        # ----------构建特征向量----------
        features = []
        # 缺失特征时使用的兜底默认值
        defaults = {
            "age": 65.0,
            "gender_m": 0,
            "admission_type_emergency": 0,
            "los_days": 5.0,
            "diagnosis_count": 3.0,
            "has_diabetes": 0,
            "has_heart_failure": 0,
            "has_hypertension": 0,
            "has_renal_disease": 0,
            "has_pneumonia": 0,
            "has_sepsis": 0,
            "has_copd": 0,
            "num_icu_stays": 0,
            "total_icu_los": 0.0,
            "max_icu_los": 0.0,
        }

        # 严格按照训练时FEATURE_COLS的顺序依次取出特征，缺省则填充默认值
        for col in FEATURE_COLS:
            features.append(float(patient_profile.get(col, defaults[col])))

        # 转为numpy数组；reshape(1,-1) 把一维列表变为【1行 N列】的二维数组，适配模型输入格式
        feature_array = np.array(features).reshape(1, -1)
        # predict_proba返回 [不发生再入院概率, 发生再入院概率]，取索引1拿到再入院风险分 0~1
        risk_score = float(self.model.predict_proba(feature_array)[0][1])

        # ----------风险分层阈值划分----------
        if risk_score >= 0.6:
            risk_level = "HIGH"
            interpretation = (
                f"High 30-day readmission risk ({risk_score:.1%}). "
                "Consider enhanced discharge planning, early follow-up appointment, "
                "and patient education on warning signs."
            )
        elif risk_score >= 0.3:
            risk_level = "MODERATE"
            interpretation = (
                f"Moderate 30-day readmission risk ({risk_score:.1%}). "
                "Standard discharge planning with scheduled follow-up recommended."
            )
        else:
            risk_level = "LOW"
            interpretation = (
                f"Low 30-day readmission risk ({risk_score:.1%}). "
                "Routine discharge process appropriate."
            )

        logger.info("risk_prediction", level=risk_level, score=risk_score)
        # 返回最终预测结果
        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "interpretation": interpretation,
            "features_used": patient_profile,
        }


def extract_patient_profile_from_query(question: str) -> dict:
    """
    简易关键词提取器：从用户自然语言提问文本里自动抓取病人相关特征信息
    :param question: 用户原始自然语言问句
    :return: 提取出来的病人参数字典（未提取到的字段不会出现在字典，后续predict函数自动补默认值）
    """
    profile = {}
    # 全部转为小写，实现大小写无关匹配
    question_lower = question.lower()

    # 正则匹配年龄：例如 "72 year old"
    import re
    age_match = re.search(r"(\d+)[- ]?year[s]?[- ]?old", question_lower)
    if age_match:
        profile["age"] = float(age_match.group(1))

    # 性别关键词匹配；gender_m=1代表男性，0代表女性
    # 用词边界 \b 避免 "male" 误匹配 "female"（female 里的 male 前有 fe，非词边界）
    if re.search(r"\bmale\b", question_lower) or any(w in question_lower for w in [" man ", "mr.", "his "]):
        profile["gender_m"] = 1
    elif re.search(r"\bfemale\b", question_lower) or any(w in question_lower for w in ["woman", "mrs.", "her "]):
        profile["gender_m"] = 0

    # 判断是否急诊入院
    if "emergency" in question_lower or "urgent" in question_lower:
        profile["admission_type_emergency"] = 1

    # 疾病特征关键词映射表：特征名 -> 相关关键词列表
    condition_map = {
        "has_diabetes": ["diabetes", "diabetic", "hyperglycemia"],
        "has_heart_failure": ["heart failure", "chf", "cardiac failure"],
        "has_hypertension": ["hypertension", "high blood pressure"],
        "has_renal_disease": ["renal", "kidney disease", "ckd", "chronic kidney"],
        "has_pneumonia": ["pneumonia"],
        "has_sepsis": ["sepsis", "septic"],
        "has_copd": ["copd", "emphysema", "chronic obstructive"],
    }

    # 循环检查每种疾病关键词是否存在于问句
    for feature, keywords in condition_map.items():
        if any(kw in question_lower for kw in keywords):
            profile[feature] = 1

    # 正则匹配住院天数：必须出现住院语境（stay/admission/hospitaliz），
    # 避免把门诊症状持续时间（"cough for 3 days"）误判成住院天数。
    # 兼容 "5 day stay" / "5-day admission" / "admission for 5 days" 三种写法。
    los_match = re.search(
        r"(?:(\d+)[- ]?days?\s+(?:stay|admission|hospitali\w*)"
        r"|(?:stay|admission|hospitali\w*)\s+(?:for\s+)?(\d+)[- ]?days?)",
        question_lower,
    )
    if los_match:
        profile["los_days"] = float(los_match.group(1) or los_match.group(2))

    return profile
