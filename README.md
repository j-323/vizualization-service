# Visual Agent

Orchestrator for 5 text-to-image/video models via API and локальные.  
- FastAPI HTTP-сервис  
- LangChain Agent для динамического выбора модели  
- Celery-фоновые задачи для тяжёлых видео  
- Redis-кеш и Prometheus-метрики  
- Docker, Kubernetes, CI

## Quickstart

```bash
cp .env.example .env
scripts/setup_dev_env.sh
scripts/run_agent.sh