from __future__ import annotations

from app.db.nutrition_sync import synced_nutrition_summary
from app.db.profile_repository import get_profile


def build_nutrition_plan(user_id: str = "demo-user") -> dict:
    profile = get_profile(user_id)
    summary = synced_nutrition_summary(user_id, days=7)
    goal = str(profile.get("goal", "保持健康"))

    if "增肌" in goal or "增重" in goal:
        meals = [
            {"meal_type": "早餐", "suggestion": "燕麦、鸡蛋、牛奶或无糖酸奶，加一份水果", "focus": "蛋白质 + 碳水"},
            {"meal_type": "午餐", "suggestion": "鸡胸肉/牛肉/鱼肉，米饭或土豆，搭配两份蔬菜", "focus": "完整正餐"},
            {"meal_type": "训练前加餐", "suggestion": "香蕉加酸奶，或全麦面包加低脂奶", "focus": "训练前能量"},
            {"meal_type": "晚餐", "suggestion": "瘦肉或豆制品，主食适量，搭配蔬菜", "focus": "恢复与饱腹"},
        ]
    elif any(term in goal for term in ("减脂", "减重", "瘦身")):
        meals = [
            {"meal_type": "早餐", "suggestion": "鸡蛋、无糖酸奶、全麦主食和一份水果", "focus": "高蛋白"},
            {"meal_type": "午餐", "suggestion": "一掌心瘦肉或鱼，半盘蔬菜，适量米饭", "focus": "控制能量密度"},
            {"meal_type": "加餐", "suggestion": "无糖酸奶或一小把坚果", "focus": "避免过度饥饿"},
            {"meal_type": "晚餐", "suggestion": "豆腐、鱼虾或鸡肉，搭配大量蔬菜和少量主食", "focus": "规律进餐"},
        ]
    else:
        meals = [
            {"meal_type": "早餐", "suggestion": "鸡蛋、牛奶或豆浆，搭配全麦主食和水果", "focus": "均衡开始"},
            {"meal_type": "午餐", "suggestion": "一份优质蛋白、主食和两份蔬菜", "focus": "均衡正餐"},
            {"meal_type": "加餐", "suggestion": "水果、酸奶或少量坚果", "focus": "按饥饿程度选择"},
            {"meal_type": "晚餐", "suggestion": "鱼肉、鸡肉或豆制品，搭配蔬菜和适量主食", "focus": "清淡收尾"},
        ]

    average = summary["daily_average"]
    food_entries = int(summary.get("food_log", {}).get("entries", 0))
    sync_note = f"已同步最近7天的 {food_entries} 条精确食物记录。" if food_entries else "最近7天暂未找到精确食物记录，先提供通用方案。"
    if summary.get("target_ready"):
        sync_note += f" 日均摄入约 {average['calories']} kcal、蛋白质 {average['protein_g']} g；计划会优先补足与目标的差额。"

    return {
        "user_id": user_id,
        "goal": goal,
        "targets": {
            "calories_target": summary.get("calories_target"),
            "protein_target_g": summary.get("protein_target_g"),
            "carbs_target_g": summary.get("carbs_target_g"),
            "fat_target_g": summary.get("fat_target_g"),
            "target_ready": summary.get("target_ready", False),
        },
        "recent_intake": {
            "days": 7,
            "entries": summary.get("entries", 0),
            "food_entries": food_entries,
            "totals": {key: summary[key] for key in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")},
            "daily_average": average,
            "target_gap_daily": summary.get("target_gap_daily", {}),
            "source": summary.get("sync_source"),
        },
        "meals": meals,
        "guidance": sync_note + " 这是通用规划，不替代医生或注册营养师建议；有慢性病、过敏或特殊饮食需求时请进一步调整。",
    }

