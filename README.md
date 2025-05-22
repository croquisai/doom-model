to set up at home
install the deps that i didnt include cuz i forgot
run the dataset maker, then train and then choose your inference type... thats it

arch: (will be expanded on with more frames in context)
```
+-------------------+
|  prev_frame (B,C,H,W)
+-------------------+
          |
      Encoder
          |
+-------------------+
|  encoded_feat (B, N)
+-------------------+
          |
          |-------------------+
          |                   |
+-------------------+    +-------------------+
| action (B, D)     |    |                   |
+-------------------+    |                   |
          |              |                   |
  ActionEmbedding        |                   |
          |              |                   |
+-------------------+    |                   |
| action_emb (B, E) |<---+                   |
+-------------------+                        |
          |                                  |
+-------------------+                        |
| concat [encoded_feat, action_emb] (B, N+E) |
+-------------------+                        |
          |                                  |
      FullyConnected (Linear)                |
          |                                  |
+-------------------+                        |
| fc_out (B, 1024)  |                        |
+-------------------+                        |
          |                                  |
      Decoder (Linear, Unflatten,            |
      ConvTranspose2d, LeakyReLU, Sigmoid)   |
          |                                  |
+-------------------+                        |
|   out (B, C, H, W)                         |
+-------------------+<-----------------------+
```
