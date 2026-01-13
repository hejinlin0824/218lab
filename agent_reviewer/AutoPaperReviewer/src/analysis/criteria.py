from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ReviewCriteria:
    """
    审稿标准定义类。
    用于维护评分维度和具体的检查清单。
    """
    
    # 评分范围
    MIN_SCORE: int = 1
    MAX_SCORE: int = 5
    
    # 评分含义映射
    SCORE_MEANINGS: Dict[int, str] = None

    def __post_init__(self):
        if self.SCORE_MEANINGS is None:
            self.SCORE_MEANINGS = {
                1: "Reject (拒绝): 存在严重缺陷，无明显贡献。",
                2: "Weak Reject (弱拒): 贡献不足或存在明显逻辑/实验问题。",
                3: "Borderline (临界): 有一定价值但也有明显短板，需权衡。",
                4: "Weak Accept (弱收): 总体稳健，有一些小的瑕疵或创新性略显不足。",
                5: "Strong Accept (强收): 突破性进展，实验完美，理论扎实。"
            }

    # 必要的检查清单 (用于后续扩展自动化检查)
    CHECKLIST: List[str] = None
    
    def get_checklist(self) -> List[str]:
        if self.CHECKLIST is None:
            self.CHECKLIST = [
                "是否包含了数据集的详细描述？",
                "是否与 SOTA (State-of-the-Art) 方法进行了对比？",
                "是否进行了消融实验 (Ablation Study)？",
                "数学推导步骤是否清晰？",
                "引用格式是否规范？"
            ]
        return self.CHECKLIST

    def validate_score(self, score: int) -> bool:
        """验证分数是否在合法范围内"""
        return self.MIN_SCORE <= score <= self.MAX_SCORE