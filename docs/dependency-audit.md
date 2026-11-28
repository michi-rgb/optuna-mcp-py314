# 依存関係の監査

開発・実行時の依存関係はルートの `uv.lock` に記録しています。実行時依存関係のエクスポートは `packaging/pylock.toml` に、Windows x64 と Linux x64 向けの wheel の取得先と SHA-256 は `packaging/wheels-*.txt` に記録しています。

`uv.lock` を更新したら、リポジトリのルートでエクスポートとマニフェストを再生成します。

```powershell
uv export --locked --no-dev --no-emit-project --format pylock.toml -o packaging/pylock.toml
uv run --locked python scripts/export_wheel_lists.py .
```

OSV を使ってロック済みの実行時依存関係を確認するには、`uv audit` に対応した uv で次のコマンドを実行します。

```powershell
uv audit --locked --no-dev --python-version 3.14 --python-platform x86_64-pc-windows-msvc
uv audit --locked --no-dev --python-version 3.14 --python-platform x86_64-unknown-linux-gnu
```

この監査の対象は Python パッケージです。OS、CPython、uv、Chrome／Chromium と、wheel に同梱されたネイティブライブラリなどは、必要に応じて個別に確認してください。
