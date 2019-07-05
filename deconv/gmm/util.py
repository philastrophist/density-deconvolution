import torch


def k_means(X, k, max_iters=50, tol=1e-9, device=None):

        n, d = X.shape

        x_min = torch.min(X, dim=0)[0]
        x_max = torch.max(X, dim=0)[0]

        resp = torch.zeros(n, k, dtype=torch.bool, device=device)
        idx = torch.arange(n)

        centroids = torch.rand(
            k, d, device=device
        ) * (x_max - x_min) + x_min

        prev_distance = torch.tensor(float('inf'), device=device)

        for i in range(max_iters):
            distances = (X[:, None, :] - centroids[None, :, :]).norm(dim=2)
            labels = distances.min(dim=1)[1]
            for j in range(k):
                centroids[j, :] = X[labels == j, :].mean(dim=0)
            resp[:] = False
            resp[idx, labels] = True
            total_distance = distances[resp].sum()

            if torch.abs(total_distance - prev_distance) < tol:
                break
            prev_distance = total_distance

        return resp.float(), centroids