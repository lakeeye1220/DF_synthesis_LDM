best_image = imagenet_inversion.main()
torch.save(best_image, 'best_image0513.pt')
best_image = torch.load('/home/hyunsoo/inversion/DF_synthesis_LDM/best_image0513.pt', weights_only=True)