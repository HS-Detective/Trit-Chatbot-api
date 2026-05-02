import pytest
from features.nav.utils import extract_country

@pytest.mark.parametrize("query, expected_country", [
    ("베트남의 주요 수출 품목은 무엇인가요?", "베트남"),
    ("독일 관련 통계는 어디서 보나요?", "독일"),
    ("수출입 통계에 대해 알려줘", None),
    ("대한민국은 어떤가요?", None),
])
def test_extract_country(query, expected_country):
    """extract_country 함수가 쿼리에서 국가명을 정확히 추출하는지 테스트합니다."""
    assert extract_country(query) == expected_country