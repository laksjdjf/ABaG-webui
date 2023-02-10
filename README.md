# ABaG-webui
このリポジトリはbirdman氏考案の構図操作手法、[ABaG](https://github.com/birdManIkioiShota/ABaG)を[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)上で実装するコードです。

# 使い方
インストールは他の拡張と同じです。インストールしたらscriptsに"ABaG for webui"という項目がでてきます。

そしたらconfigファイル(stable-diffusion-webui/configs/v1-inference.yaml)をリポジトリにあるものに差し替えてください(もしくは use_checkpoint: Falseにする)。gradient checkpointingを無効化しただけなので通常の画像生成に影響しない・・と思います。

samplerはDDIMにしてください、それ以外のsamplerでは適応されません。UIの通り、Enableにチェックを入れ、bboxをbirdman氏のコードと同様の記法で入力します。学習率を数字で入力してください。set size of attention map toは変更しない方がいいです。
