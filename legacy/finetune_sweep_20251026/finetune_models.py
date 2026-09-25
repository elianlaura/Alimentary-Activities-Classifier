import os

fine_tune_models = {
    4: "saved_models_prev/recurrent_models_20251011-123315/de_fake_padts_94u_1f_autoencoder_s_20251011-123937/best_model_de_fake_padts_94u_20251011-123937.keras",
    13: "saved_models_prev/recurrent_models_20251012-100533/de_fake_padts_94u_1f_autoencoder_s_20251012-101512/best_model_de_fake_padts_94u_20251012-101512.keras",
    14: "saved_models_prev/recurrent_models_20251019-194221/de_d_e_fake_100-9_1f_autoencoder_s_20251020-050022/best_model_de_d_e_fake_100-9_20251020-050022.keras",
    16: "saved_models_prev/recurrent_models_20251019-194221/de_d_e_fake_100-9_1f_autoencoder_s_20251019-194257/best_model_de_d_e_fake_100-9_20251019-194257.keras",
    18: "saved_models_prev/recurrent_models_20251019-194221/de_d_e_fake_100-9_1f_autoencoder_s_20251020-104835/best_model_de_d_e_fake_100-9_20251020-104835.keras",
    20: "saved_models_prev/recurrent_models_20251019-194313/de_d_e_fake_100-9_1f_autoencoder_s_20251019-194352/best_model_de_d_e_fake_100-9_20251019-194352.keras",
    21: "saved_models_prev/recurrent_models_20251019-194313/de_d_e_fake_100-9_1f_autoencoder_s_20251020-050942/best_model_de_d_e_fake_100-9_20251020-050942.keras",
    23: "saved_models_prev/recurrent_models_20251019-194313/de_d_e_fake_100-9_1f_autoencoder_s_20251020-012946/best_model_de_d_e_fake_100-9_20251020-012946.keras",
    24: "saved_models_prev/recurrent_models_20251019-194313/de_d_e_fake_100-9_1f_autoencoder_20251020-075710/best_model_de_d_e_fake_100-9_20251020-075710.keras",
    25: "saved_models_prev/recurrent_models_20251019-194313/de_d_e_fake_100-9_1f_autoencoder_20251020-105446/best_model_de_d_e_fake_100-9_20251020-105446.keras",
    15: "saved_models_prev/recurrent_models_20251019-194221/de_d_e_fake_100-9_1f_autoencoder_20251020-230730/best_model_de_d_e_fake_100-9_20251020-230730.keras",
    17: "saved_models_prev/recurrent_models_20251019-194221/de_d_e_fake_100-9_1f_autoencoder_20251020-193901/best_model_de_d_e_fake_100-9_20251020-193901.keras",
    19: "saved_models_prev/recurrent_models_20251019-194221/de_d_e_fake_100-9_1f_autoencoder_20251020-150955/best_model_de_d_e_fake_100-9_20251020-150955.keras",
    22: "saved_models_prev/recurrent_models_20251019-194313/de_d_e_fake_100-9_1f_autoencoder_20251020-131801/best_model_de_d_e_fake_100-9_20251020-131801.keras",
    23: "saved_models/recurrent_models_20260407-095317/de_rec_fake_94u_concat3_1f_autoencoder_s_20260407-095328/best_model_de_rec_fake_94u_concat3_20260407-095328.keras",
    24: "saved_models/recurrent_models_20260413-154549/de_rec_fake_94u_concat3d_1f_autoencoder_s_20260413-154620/best_model_de_rec_fake_94u_concat3d_20260413-154620.keras",
    25: "saved_models/recurrent_models_20260422-124718/de_rec_fake_94u_concat3d_1f_autoencoder_s_20260422-124746/best_model_de_rec_fake_94u_concat3d_20260422-124746.keras",
    26: "saved_models/recurrent_models_20260423-125940/de_rec_fake_94u_concat4d_1f_autoencoder_s_20260423-130033/best_model_de_rec_fake_94u_concat4d_20260423-130033.keras",
    27: "saved_models/recurrent_models_20260423-135800/de_rec_fake_94u_concat4d_1f_autoencoder_s_20260423-135856/best_model_de_rec_fake_94u_concat4d_20260423-135856.keras",
}
