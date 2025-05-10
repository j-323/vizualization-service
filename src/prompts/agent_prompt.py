# src/prompts/agent_prompt.py

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# 1. Системное сообщение: объясняем роль агента и требование по комплаенсу
SYSTEM_MESSAGE = """\
Вы — интеллектуальный агент, который оркестрирует пять инструментов генерации изображений и видео.
Перед тем, как выбрать инструмент, вы **обязательно** проверяете пользовательский запрос на соответствие
законам и этическим нормам: 
- Если запрос нарушает закон или этические правила, откажитесь с кратким объяснением.
- Если запрос корректен, выберите наиболее подходящий инструмент и сформируйте для него запрос в виде: 

<Имя_инструмента>
<Текстовая_подсказка_для_этого_инструмента>
"""

# 2. Шаблон для пользовательского сообщения
USER_TEMPLATE = """\
Пользователь запросил:
{user_request}
Дайте ответ в формате указанном в системном сообщении.
"""

# 3. Собираем ChatPromptTemplate
agent_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE),
    HumanMessagePromptTemplate.from_template(USER_TEMPLATE),
])

def build_agent_prompt(user_request: str) -> str:
    """
    Формирует финальный промпт для LangChain-агента.
    """
    return agent_prompt.format_prompt(user_request=user_request).to_string()
