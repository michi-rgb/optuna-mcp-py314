# Optuna MCP for Python 3.14

Optuna で実験条件を提案し、評価結果を記録・可視化するための MCP サーバーです。[Optuna 公式 MCP サーバー](https://github.com/optuna/optuna-mcp/tree/c59e6340c7630eecaac77b292bf2e291b7671261) をベースにした独立した改変版です。

- CPython 3.14 の Windows x64 と Linux x64 を依存関係の解決対象にしています。
- CPU 版 PyTorch を使う `GPSampler` が既定です。`ask_startup_trials` で初期試行をまとめて提案できます。
- MCP クライアントとは標準入出力で接続します。試行結果はメモリまたは SQLite に保存できます。
- PNG の生成には Kaleido と Chrome または Chromium を使います。Optuna Dashboard はループバックアドレスで起動します。

## GitHub から直接起動する

Windows x64 または Linux x64 で、Codex が実行される環境の PATH に Git と uv（`uvx`）を用意します。`uvx --help` で `--torch-backend` が使えることを確認してください（uv 0.13.6 の CLI で確認済み）。

[設定例](examples/codex/config.toml) の `[mcp_servers.optuna_py314]` ブロックを、Codex のユーザー設定 `~/.codex/config.toml` に追加して Codex を再起動します。同じ名前のブロックがすでにある場合は、二重に追加せず既存のブロックを編集してください。

事前のクローンや ZIP の展開は不要です。初回は `uvx` が GitHub の指定コミットからパッケージを取得し、CPython 3.14 がなければ取得します。CPU 版 PyTorch などの依存関係も取得するため、ネット接続が必要で、初回は起動に時間がかかります。

ソースはコミット SHA で固定します。依存バージョンは `uvx` が実行環境を作る際にパッケージの依存条件から解決します。`uv.lock` のバージョンで起動したい場合は、次の手順を使ってください。

## クローンしてロック済みの依存関係で起動する

uv が使える環境で、このリポジトリのルートから実行します。

```powershell
uv python install 3.14
uv sync --locked --python 3.14
uv run --locked optuna-mcp --version
```

Windows の Codex CLI に登録する例です。別のディレクトリからでも同じソースを起動できるように、プロジェクトの絶対パスを渡します。

```powershell
$project = (Get-Location).Path
codex mcp add optuna_py314 -- uv --directory $project run --locked optuna-mcp
```

SQLite に保存する場合は、登録コマンドの末尾に `--storage sqlite:///optuna.db` を追加します。プロジェクトのルートに `optuna.db` が作成されます。

## 最適化の例

MCP クライアントに「`demo` という study で、−5 から 5 の範囲で実数 `x` を変えながら `x²` を最小化して」と指示します。ツールを直接使う場合は次の順で進めます。

1. `create_study` で `directions=["minimize"]` を指定します。
2. `ask_startup_trials` に探索範囲を渡して初期条件を取得します。初期試行数は既定で10件です。`create_study` の `n_startup_trials` で変更できます。
3. 各条件を評価し、試行番号と目的値を `tell` で登録します。
4. 初期試行を終えたら `ask` で次の条件を提案し、`best_trial` で最良の試行を確認します。

## ビルドと検証

```powershell
uv build --wheel --out-dir wheels
uv run --locked --no-sync python scripts/prepare_wheelhouse.py
uv run --locked --no-sync python scripts/smoke_test.py
```

`prepare_wheelhouse.py` は対象環境向けの依存 wheel を取得し、`packaging/wheels-*.txt` に記載された SHA-256 と照合します。スモークテストはローカルで生成した wheel をオフラインで起動します。PNG も確認する場合は、Chrome または Chromium を用意し、スモークテストに `--plot` を追加します。[依存関係の監査手順](docs/dependency-audit.md)

## ライセンス

MIT ライセンスです。上流の Preferred Networks, Inc. の著作権表示と許諾文は [LICENSE](LICENSE) に記載しています。
