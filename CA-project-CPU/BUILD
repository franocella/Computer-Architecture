# Contenuto del BUILD principale

cc_binary(
    name = "Preprocessing_opt",
    srcs = ["Preprocessing_opt.cpp"],
    deps = [
        # Dipendenza Abseil
        "@abseil-cpp//absl/strings",
        # Dipendenza RE2 (VERIFICA WORKSPACE SE CI SONO ANCORA ERRORI LNK)
        "@re2//:re2",
    ],
    features = ["c++17"]

)

# --- Target per la versione NON ottimizzata (SENZA RE2) ---
cc_binary(
    name = "Preprocessing_no_opt",
    srcs = ["Preprocessing_no_opt.cpp"],

    # Dipendenze per questa versione
    deps = [],
    features = ["c++17"]
)