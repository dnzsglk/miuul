pca3d = PCA(n_components=3)
    comps = pca3d.fit_transform(X_scaled)

    df_pca3d = pd.DataFrame(comps, columns=["PC1", "PC2", "PC3"])
    df_pca3d["Cluster"] = clusters

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        df_pca3d["PC1"],
        df_pca3d["PC2"],
        df_pca3d["PC3"],
        c=df_pca3d["Cluster"],
        cmap="viridis",
        s=70,
        alpha=0.8
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Müşteri Segmentleri (PCA 3D - K={optimal_k})")

    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)

    plt.show()
    plt.close()
