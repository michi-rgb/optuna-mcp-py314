# 依存関係の監査

開発・実行時の依存関係は `uv.lock` に、実行時依存関係のエクスポートは `pylock.toml` に記録しています。Windows x64 と Linux x64 向けの wheel の取得先と SHA-256 は `wheels-*.txt` に記録しています。

OSV を使ってロック済みの実行時依存関係を確認するには、`uv audit` に対応した uv で次のコマンドを実行します。

```powershell
uv audit --locked --no-dev --python-version 3.14 --python-platform x86_64-pc-windows-msvc
uv audit --locked --no-dev --python-version 3.14 --python-platform x86_64-unknown-linux-gnu
```

この監査の対象は Python パッケージです。OS、CPython、uv、Chrome／Chromium と、wheel に同梱されたネイティブライブラリなどは、必要に応じて個別に確認してください。
