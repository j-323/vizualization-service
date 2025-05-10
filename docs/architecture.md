
---

### docs/architecture.md

```markdown
# Архитектурный обзор

- **FastAPI**: HTTP-эндпоинты `/generate`  
- **LangChain Agent**: выбор между 5 инструментами  
- **Celery**: асинхронная генерация (особенно видео)  
- **Redis**: кеш ответов и брокер Celery  
- **Prometheus**: метрики latency, success/fail  
- **Kubernetes**: продакшн-деплой (Deployment, Service, Ingress)