import json
import logging
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from src.config.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Загружаем правила комплаенса из JSON-конфига
try:
    with open(settings.COMPLIANCE_POLICIES_PATH, encoding="utf-8") as f:
        policies = json.load(f)
except Exception as e:
    logger.error("Не удалось загрузить политики комплаенса", exc_info=e)
    policies = {"banned_categories": []}

# Системное сообщение с динамичной вставкой запрещённых категорий
SYSTEM_MESSAGE = f"""\
Вы — интеллектуальный агент, оркестрирующий пять инструментов генерации.
Перед выполнением запроса обязательно проверяйте его на соответствие:
- Запрещённый контент: {', '.join(policies['banned_categories'])}.
Если запрос нарушает нормы — откажите с объяснением, иначе выберите инструмент и верните:

<Имя_инструмента>
<Точный промпт для этого инструмента>
"""

# Загружаем few-shot примеры из внешнего файла
examples = []
try:
    with open(settings.FEWSHOT_EXAMPLES_PATH, encoding="utf-8") as f:
        examples = json.load(f)
except Exception as e:
    logger.warning("Few-shot примеры не загружены", exc_info=e)

# Строим шаблоны сообщений
messages = [SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE)]
for ex in examples:
    messages.append(HumanMessagePromptTemplate.from_template(ex["user"]))
    messages.append(AIMessagePromptTemplate.from_template(ex["assistant"]))
# Финальное место для реального запроса
messages.append(HumanMessagePromptTemplate.from_template("{user_request}"))

agent_prompt = ChatPromptTemplate.from_messages(messages)

def build_agent_prompt(user_request: str) -> str:
    """
    Формирует финальный систем+few-shot+user промпт,
    логирует ошибки сборки.
    """
    try:
        return agent_prompt.format_prompt(user_request=user_request).to_string()
    except Exception as e:
        logger.error("Ошибка при формировании промпта агента", exc_info=e)
        # в крайнем случае возвращаем базовый системный + user
        fallback = SYSTEM_MESSAGE + "\n\nПользователь запросил:\n" + user_request
        return fallback