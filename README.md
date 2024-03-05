# Reference

Please refer to https://inaho.esa.io/posts/495/edit#10 if you want to know how to use this soft.


# 擬似的な葉面積密度指数を計算するための画像処理スクリプト

## 使用方法

```
./main.py INPUT_PATH [オプション]
```

`INPUT_PATH`には、処理対象の画像ファイルへのパスを指定してください。

### オプション

- `--conf-path PATH`: 設定ファイル（TOML形式）へのパスを指定します。デフォルト値は`./conf/conf.toml`です。
- `--output-dir PATH`: 出力画像を保存するディレクトリのパスを指定します。デフォルト値は`./data/output`です。ディレクトリが存在しない場合は生成します。
- `--hmin INTEGER`: 画像の切り取り開始高さを指定します。省略した場合のデフォルト値は0です。
- `--hmax INTEGER`: 画像の切り取り終了高さを指定します。省略すると自動的に計算されます。
- `--help`: ヘルプメッセージを表示し、終了します。

### 使用例

以下のコマンド例では、サンプル画像に対して、切り取り開始高さを`0`、切り取り終了高さを`2500`と指定した形で結果画像を生成します。

```
./main.py data/ikeuchi_anno_data/sample_images/20211005_132152.jpg --hmin 0 --hmax 2500
```