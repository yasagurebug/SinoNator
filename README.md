# SinoNator

はい/いいえの質問に答えると、配信アーカイブを1本おすすめする Next.js アプリです。

## Requirements

- Node.js 18+
- Python 3.11+（質問データ再生成を行う場合）

## Local Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## データ更新（質問セット再生成）

`public/data/runtime_v2.json` を作り直すと、UI/API の参照データが更新されます。

```bash
export OPENAI_API_KEY="sk-xxxx"
python3 scripts/v2_run_pipeline.py --set-id 2026-02 --publish
```

- 監査ゲート（StepB/D/E/F）で一時停止します。
- 監査完了後は Enter（または `yes`）で続行します。
- `data/v2/questions_v2.json` を手修正した後は StepF を再実行します。

```bash
python3 scripts/v2_run_pipeline.py --set-id 2026-02 --from-step F --publish
```

## Deploy

GitHub に push し、Vercel の自動ビルド/デプロイで公開します。

## License

MIT (`LICENSE`)
