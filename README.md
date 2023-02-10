# ABaG-webui
このリポジトリはbirdman氏考案の構図操作手法、[ABaG](https://github.com/birdManIkioiShota/ABaG)を[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)上で実装するコードです。

# 使い方
インストールは他の拡張と同じです。インストールしたらscriptsに"ABaG for webui"という項目がでてきます。

そしたら **configファイル(stable-diffusion-webui/configs/v1-inference.yaml)をリポジトリにあるものに差し替えてください(もしくは use_checkpoint: Falseにする) 。** gradient checkpointingを無効化しただけなので通常の画像生成に影響しない・・と思います。

--no-half --precision fullで起動してください。fp16でもうまくいくこともあるのですが、loss=nanになることもあります。

samplerは **DDIM** にしてください、それ以外のsamplerでは適応されません。UIの通り、Enableにチェックを入れ、bboxをbirdman氏のコードと同様の記法で入力します。そしてlrを設定してください。lrといっても学習してるわけではなく、この手法の効き目をどれくらい強くするかという設定です。set size of attention map toは変更しない方がいいです。

sampling stepは50を想定しているっぽいような実装っぽいのでそうしたほうがいいっぽいです。

# 注意点
+ 動いたから公開しちゃお、程度です。
+ 微分計算を何度も繰り返すので、生成は遅いし、VRAM必要量もあがります。
+ フィーリングで実装したので元のbirdman氏のアルゴリズムと違いがあるかもしれません（ていうかある）
+ おそらくHypernetworksおよびxformersは一部のモジュールで適用されなくなります。(LoRAは適用されると思う・・・)
+ v1でもv2(yamlでuse_checkpoint:Falseにする)でも動くようですが、意図したとおりになっているか分かりません。
+ txt2imgしか想定していません。
+ モジュールを書き換えるため、別の機能に何か意図しないことが起こるかもしれません。

# todo（いい方法教えて）
+ bboxの指定がCUI的でこれじゃGUIの意味がない
+ トークンIDがよくわからない
+ lossの表示でプログレスバーがしぬ
+ fp16対応
# 既知の問題
+ しらん

# 引用リポジトリ
https://github.com/birdManIkioiShota/ABaG ：本家ABaG

https://github.com/AttendAndExcite/Attend-and-Excite ：本家の本家

https://github.com/Stability-AI/stablediffusion ：本家の本家の本家の・・・

https://github.com/kousw/stable-diffusion-webui-daam ：Attention Mapの取り出し方を参考にしました。
