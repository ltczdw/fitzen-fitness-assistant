from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class AuthoritySeed:
    name: str
    url: str
    topics: tuple[str, ...]


ALLOWED_AUTHORITY_DOMAINS: frozenset[str] = frozenset(
    {
        "who.int",
        "cdc.gov",
        "nhs.uk",
        "health.gov",
        "nih.gov",
        "ncbi.nlm.nih.gov",
        "cnsoc.org",
        "nhc.gov.cn",
        "gov.cn",
        "acsm.org",
    }
)


def is_authority_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in ALLOWED_AUTHORITY_DOMAINS)


AUTHORITY_SEEDS: tuple[AuthoritySeed, ...] = (
    AuthoritySeed(
        "WHO physical activity guideline",
        "https://www.who.int/publications/i/item/9789240014886",
        ("运动学", "训练", "康复", "运动安全"),
    ),
    AuthoritySeed(
        "WHO healthy diet",
        "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        ("营养学", "膳食", "运动饮食"),
    ),
    AuthoritySeed(
        "CDC physical activity benefits",
        "https://www.cdc.gov/physical-activity-basics/health-benefits/adults.html",
        ("运动学", "健康", "体重管理"),
    ),
    AuthoritySeed(
        "ODPHP physical activity guidelines",
        "https://odphp.health.gov/our-work/nutrition-physical-activity/physical-activity-guidelines/current-guidelines",
        ("训练", "运动处方", "运动安全"),
    ),
    AuthoritySeed(
        "NHS sprains and strains",
        "https://www.nhs.uk/conditions/sprains-and-strains/",
        ("康复学", "损伤", "运动安全"),
    ),
    AuthoritySeed(
        "NCBI exercise physiology",
        "https://www.ncbi.nlm.nih.gov/books/NBK482280/",
        ("运动学", "运动生理学", "恢复"),
    ),
    AuthoritySeed(
        "Chinese Society of Nutrition dietary guide",
        "https://www.cnsoc.org/bookpublica/0522202019.html",
        ("营养学", "膳食", "运动饮食"),
    ),
)
