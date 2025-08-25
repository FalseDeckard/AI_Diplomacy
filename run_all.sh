MODEL="together:openai/gpt-oss-120b"

for folder in results/*; do
  if [ -d "$folder" ]; then
    echo "🔍 Running analyze_lies.py for $folder"
    python3 analyze_lies.py "$folder" \
      --model "$MODEL" \
      --output "$folder/lie_report.md" \
      --json "$folder/lie_results.json"
    echo "✅ Finished $folder"
    echo "-----------------------------------"
  fi
done