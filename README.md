# ABaG-webui
このリポジトリはbirdman氏考案の構図操作手法、[ABaG](https://github.com/birdManIkioiShota/ABaG)を[webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)上で実装するコードです。

# 使い方
インストールは他の拡張と同じです。インストールしたらscriptsに"ABaG for webui"という項目がでてきます。

+ yamlファイルで **use_checkpoint: False** にしてください。v1の場合はたぶんstable-diffusion-webui/configs/v1-inference.yamlです。gradient checkpointingを無効化しただけなので通常の画像生成に影響しない・・と思います。

+ **--no-half --precision full** で起動してください。fp16でもうまくいくこともあるのですが、loss=nanになることもあります。

+ samplerは **DDIM** にしてください、それ以外のsamplerでは適応されません。
+ Enableにチェックを入れてください。
+ bboxをbirdman氏のコードと同様の記法で入力します。AttentionMap(AIが画像のどの部分に注意を向けているかを示す配列)のサイズは(H/32,W/32)になります。512×512では(16,16)です。bboxは"<トークンID> <上> <下> <左> <右>"と指定します。"2 0 16 0 7"と指定すると2番目のトークンが左側に生成されるようになります。
+ lrを設定してください。lrといっても学習してるわけではなく、この手法の効き目をどれくらい強くするかという設定です。
+ set size of attention map toは変更しない方がいいです。
+ thresholdsは特定のステップでのlossの閾値（1-nが閾値）になります。

+ sampling stepは50を想定しているっぽいような実装っぽいのでそうしたほうがいいっぽいです。

使用例：元のリポジトリと同様左半分にねずみ、右下に赤い車が生成されるように指定してみた。

![4](https://github.com/laksjdjf/ABaG-webui/blob/main/image1.png?raw=true)

# 注意点
+ 動いたから公開しちゃお、程度です。
+ 微分計算を何度も繰り返すので、生成は遅いし、VRAM必要量もあがります。
+ フィーリングで実装したので元のbirdman氏のアルゴリズムと違いがあるかもしれません（ていうかある）
+ おそらくHypernetworksおよびxformersは一部のモジュールで適用されなくなります。(LoRAは適用されると思う・・・)
+ v1でもv2(yamlでuse_checkpoint:Falseにする)でも動くようですが、意図したとおりになっているか分かりません。
+ txt2imgしか想定していません。
+ モジュールを書き換えるため、別の機能に何か意図しないことが起こるかもしれません。
+ 外から潜在変数を無理やり置き換えるのできもい画像がでやすいです。

# todo（いい方法教えて）
+ bboxの指定がCUI的でこれじゃGUIの意味がないので変える
+ トークンIDがわかるようにする
+ lossの表示でプログレスバーがしぬ
+ fp16対応

# 既知の問題
+ しらん

# 引用リポジトリ
https://github.com/birdManIkioiShota/ABaG ：本家ABaG

https://github.com/AttendAndExcite/Attend-and-Excite ：本家の本家

https://github.com/Stability-AI/stablediffusion ：本家の本家の本家の・・・

https://github.com/kousw/stable-diffusion-webui-daam ：Attention Mapの取り出し方を参考にしました。
