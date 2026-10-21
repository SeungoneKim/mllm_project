# Steps 1-2 on one GPU. With no argument: decompose the questions (Qwen3-8B), embed the pages and score all queries
# (ColQwen2.5). With an argument, only that part: pages | decompose | score.
set -e
case "${1:-all}" in
  pages) $PY -u embed.py pages ;;
  decompose) $PY -u decompose.py ;;
  score) $PY -u embed.py score ;;
  all) $PY -u decompose.py; $PY -u embed.py pages; $PY -u embed.py score ;;
esac
