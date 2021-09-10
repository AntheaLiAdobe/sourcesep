    with torch.autograd.set_detect_anomaly(True):
        optimizer = network.configure_optimizers()

        for epoch in range(0, args.epochs):
            for batch_idx, batch in enumerate(train_loader):

                # train step
                loss = network(batch, batch_idx)

                # clear gradients
                optimizer.zero_grad()

                # backward
                loss.backward()

                # update parameters
                optimizer.step()