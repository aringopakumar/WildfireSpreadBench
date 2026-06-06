    """
    ResBlock with cross-attention conditioning (replaces additive injection).

    The conditioning image features are attended to via CrossAttentionConditioner
    and added to the hidden state after the second conv, before the skip connection.
    The timestep embedding is still injected additively (standard practice).
    """
    def __init__(self, in_ch, out_ch, t_ch, cond_ch, dropout, n_heads=4, head_dim=32):
        super(ResBlock, self).__init__()

        self.conv_1 = nn.Sequential(
            group_norm_layer(in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )
        self.time_embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_ch, out_ch)
        )
        # Cross-attention conditioning (replaces condition_conv + additive inject)
        self.cross_attn = CrossAttentionConditioner(
            hidden_ch=out_ch,
            cond_ch=cond_ch,
            n_heads=n_heads,
            head_dim=head_dim,
        )
        self.conv_2 = nn.Sequential(
            group_norm_layer(out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.skip_conn = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t, cond_img, mask):
        h = self.conv_1(x)
        # Timestep: additive (standard)
        h = h + self.time_embedding(t)[:, :, None, None]
        h = self.conv_2(h)
        # Conditioning: cross-attention, CFG-masked
        h = h + self.cross_attn(h, cond_img, mask)
        return h + self.skip_conn(x)
