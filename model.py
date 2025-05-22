import torch
import torch.nn as nn

class ActionEmbedding(nn.Module):
    def __init__(self, action_dim, emb_dim):
        super().__init__()
        self.embedding = nn.Linear(action_dim, emb_dim)

    def forward(self, x):
        return self.embedding(x)

class ConditionedFrameGen(nn.Module):
    def __init__(self, img_channels=3, action_dim=3, emb_dim=32, input_size=128):
        super().__init__()
        self.action_emb = ActionEmbedding(action_dim, emb_dim)
        self.img_channels = img_channels
        self.input_size = input_size

        encoder_layers = []
        decoder_layers = []
        channels = [img_channels, 32, 64, 128, 256, 512]
        size = input_size
        # Encoder
        for i in range(len(channels) - 1):
            encoder_layers.append(nn.Conv2d(channels[i], channels[i+1], 4, 2, 1))
            encoder_layers.append(nn.LeakyReLU())
            size = size // 2
        self.encoder = nn.Sequential(*encoder_layers)
        self.enc_feat_size = channels[-1] * size * size

        # fc layer
        self.fc = nn.Linear(self.enc_feat_size + emb_dim, 1024)

        decoder_layers.append(nn.Linear(1024, self.enc_feat_size))
        decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Unflatten(1, (channels[-1], size, size)))
        for i in range(len(channels) - 1, 0, -1):
            decoder_layers.append(nn.ConvTranspose2d(channels[i], channels[i-1], 4, 2, 1))
            if i > 1:
                decoder_layers.append(nn.LeakyReLU())
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, prev_frame, action):
        # prev_frame: (B, C, H, W)
        # action: (B, action_dim)
        B = action.shape[0]
        H = self.input_size
        W = self.input_size
        if prev_frame is None:
            prev_frame = torch.zeros((B, self.img_channels, H, W), device=action.device)
        x = self.encoder(prev_frame)
        x = x.view(x.size(0), -1)
        a = self.action_emb(action)
        x = torch.cat([x, a], dim=1)
        x = self.fc(x)
        out = self.decoder(x)
        return out


"""
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

"""