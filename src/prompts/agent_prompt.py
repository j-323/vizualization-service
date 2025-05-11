# src/prompts/agent_prompt.py

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

# 1. Системное сообщение: объясняем роль агента и требование по комплаенсу
SYSTEM_MESSAGE = """\
Вы — интеллектуальный агент, который оркестрирует пять инструментов генерации изображений и видео.
Перед тем, как выбрать инструмент, вы **обязательно** проверяете пользовательский запрос на соответствие
законам, этическим нормам и политике безопасности:
- Если запрос нарушает закон или этические правила (например, пропаганда насилия, нелегальный контент),
  откажитесь с кратким объяснением: "Запрос запрещён из-за ...".
- Если всё в порядке, выберите наиболее подходящий инструмент и сформируйте для него запрос в точном формате:

<Имя_инструмента>
<Промпт для этого инструмента>
"""

# 2. Несколько демонстрационных примеров "few-shot"
EXAMPLE_1_USER = """Пользователь запросил:
Сгенерируй фотореалистичное изображение заката над горами"""
EXAMPLE_1_ASSISTANT = """ImageB_API
Фотореалистичное изображение заката над высокими снежными горами с мягким туманом"""

EXAMPLE_2_USER = """Пользователь запросил:
Создай короткую MP4-анимацию бабочки, летящей над цветами"""
EXAMPLE_2_ASSISTANT = """VideoD_Local
Короткая loop-анимация (MP4): мультяшная бабочка, летающая над полем ярких полевых цветов"""

# 3. Шаблон для реального пользовательского сообщения
USER_TEMPLATE = """Пользователь запросил:
{user_request}
Дайте ответ в формате, указанном в системном сообщении."""

# 4. Собираем ChatPromptTemplate со всеми сообщениями
agent_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template(EXAMPLE_1_USER),
    AIMessagePromptTemplate.from_template(EXAMPLE_1_ASSISTANT),
    HumanMessagePromptTemplate.from_template(EXAMPLE_2_USER),
    AIMessagePromptTemplate.from_template(EXAMPLE_2_ASSISTANT),
    HumanMessagePromptTemplate.from_template(USER_TEMPLATE),
])

def build_agent_prompt(user_request: str) -> str:
    """
    Формирует финальный промпт для LangChain-агента,
    включая проверку комплаенса и демонстрационные примеры.
    """
    return agent_prompt.format_prompt(user_request=user_request).to_string()