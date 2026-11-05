from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .food_repository import SEED_ALIASES, SEED_FOODS, SEED_PORTIONS, normalize_food_name


NUTRIENT_FIELDS = ("energy_kcal", "protein_g", "carb_g", "fat_g")
REQUIRED_COMMON_FOODS: dict[str, tuple[str, ...]] = {
    "谷薯类": ("熟大米", "熟面条", "燕麦", "馒头", "普通面包", "土豆"),
    "畜禽肉类": ("鸡胸肉", "鸡腿肉", "猪里脊肉", "瘦牛肉"),
    "鱼虾类": ("三文鱼", "虾仁", "鳕鱼"),
    "蛋类": ("鸡蛋", "鸡蛋白"),
    "乳类": ("纯牛奶", "低脂牛奶", "无糖酸奶"),
    "豆制品": ("北豆腐", "无糖豆浆"),
    "蔬菜类": ("西兰花", "菠菜", "生菜", "黄瓜", "番茄", "胡萝卜", "鲜香菇"),
    "水果类": ("香蕉", "苹果", "橙子", "梨", "葡萄", "草莓"),
    "油脂类": ("食用植物油", "橄榄油", "花生油", "芝麻油"),
}


@dataclass(frozen=True)
class CatalogIssue:
    code: str
    severity: str
    food_name: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _issue(issues: list[CatalogIssue], code: str, severity: str, food_name: str, message: str) -> None:
    issues.append(CatalogIssue(code, severity, food_name, message))


def audit_seed_catalog() -> dict[str, Any]:
    """Audit the checked-in catalog before it is seeded into PostgreSQL.

    The audit intentionally runs against the source catalog so local fallback
    mode and a newly initialized database are checked by the same rules.
    """
    issues: list[CatalogIssue] = []
    names_by_state: set[tuple[str, str]] = set()
    canonical_names: set[str] = set()

    for index, row in enumerate(SEED_FOODS):
        if len(row) != 8:
            _issue(issues, "field_count", "高", str(row[0]) if row else f"#{index}", "食材记录必须包含名称、分类、状态和4项营养字段及膳食纤维")
            continue

        food_name, category, state, energy, protein, carb, fat, fiber = row
        display_name = str(food_name)
        normalized_name = normalize_food_name(display_name)
        key = (normalized_name, normalize_food_name(str(state)))

        if not display_name.strip() or not str(category).strip() or not str(state).strip():
            _issue(issues, "identity_missing", "高", display_name, "食材名称、分类、加工状态不能为空")
        if key in names_by_state:
            _issue(issues, "duplicate_food", "高", display_name, "相同食材和加工状态重复，可能导致查询结果不确定")
        names_by_state.add(key)
        canonical_names.add(display_name)

        values = (energy, protein, carb, fat, fiber)
        if any(value is None for value in values):
            _issue(issues, "nutrient_missing", "高", display_name, "每100g营养字段存在空值")
            continue

        try:
            numeric = tuple(float(value) for value in values)
        except (TypeError, ValueError):
            _issue(issues, "nutrient_not_numeric", "高", display_name, "营养字段必须是数字")
            continue

        if not all(math.isfinite(value) for value in numeric):
            _issue(issues, "nutrient_not_finite", "高", display_name, "营养字段不能是NaN或无穷大")
            continue

        energy_value, protein_value, carb_value, fat_value, fiber_value = numeric
        if energy_value < 0 or energy_value > 1000:
            _issue(issues, "energy_out_of_range", "高", display_name, "每100g热量应在0到1000千卡之间")
        for field, value in zip(("protein_g", "carb_g", "fat_g", "fiber_g"), (protein_value, carb_value, fat_value, fiber_value)):
            if value < 0 or value > 100:
                _issue(issues, "nutrient_out_of_range", "高", display_name, f"{field} 每100g应在0到100g之间")

        macro_energy = protein_value * 4 + carb_value * 4 + fat_value * 9
        if energy_value > 0 and abs(energy_value - macro_energy) > max(25, energy_value * 0.20):
            _issue(
                issues,
                "energy_macro_mismatch",
                "中",
                display_name,
                f"热量{energy_value:g}与蛋白质/碳水/脂肪折算值{macro_energy:.1f}差异超过20%，需要复核来源或加工状态",
            )

    alias_targets: dict[str, set[str]] = {}
    for alias, target in SEED_ALIASES.items():
        normalized_alias = normalize_food_name(alias)
        alias_targets.setdefault(normalized_alias, set()).add(target)
        if target not in canonical_names:
            _issue(issues, "alias_target_missing", "高", alias, f"别名指向不存在的标准食材：{target}")

    for alias, targets in alias_targets.items():
        if len(targets) > 1:
            _issue(issues, "alias_conflict", "高", alias, f"同一个归一化别名指向多个标准食材：{sorted(targets)}")

    for target, portions in SEED_PORTIONS.items():
        if target not in canonical_names:
            _issue(issues, "portion_target_missing", "中", target, "份量参考指向不存在的标准食材")
        for portion_name, weight_g in portions.items():
            try:
                weight = float(weight_g)
            except (TypeError, ValueError):
                weight = -1
            if not math.isfinite(weight) or weight <= 0 or weight > 5000:
                _issue(issues, "portion_invalid", "中", target, f"份量“{portion_name}”的克重无效：{weight_g}")

    missing_common_foods: list[dict[str, str]] = []
    for category, required in REQUIRED_COMMON_FOODS.items():
        for food_name in required:
            if food_name not in canonical_names:
                missing_common_foods.append({"category": category, "food_name": food_name})
                _issue(issues, "missing_common_food", "中", food_name, f"{category} 高频食材尚未纳入知识库")

    summary = {
        "total_foods": len(SEED_FOODS),
        "total_aliases": len(SEED_ALIASES),
        "total_portion_foods": len(SEED_PORTIONS),
        "missing_common_foods": missing_common_foods,
        "issue_count": len(issues),
        "high_count": sum(issue.severity == "高" for issue in issues),
        "medium_count": sum(issue.severity == "中" for issue in issues),
        "low_count": sum(issue.severity == "低" for issue in issues),
    }
    return {"summary": summary, "issues": [issue.as_dict() for issue in issues]}


if __name__ == "__main__":
    import json

    print(json.dumps(audit_seed_catalog(), ensure_ascii=False, indent=2))