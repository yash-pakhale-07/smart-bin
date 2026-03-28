import os, random, shutil

TARGET = 500
classes = ['Plastic', 'Paper', 'Metal', 'Glass']

for split in ['train', 'val']:
    target = TARGET if split == 'train' else 100
    print(f'\n{split.upper()}:')
    for cls in classes:
        folder = f'data/{split}/{cls}'
        imgs = [f for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        current = len(imgs)
        print(f'  {cls}: {current}', end=' ')

        if current > target:
            random.shuffle(imgs)
            for img in imgs[target:]:
                os.remove(os.path.join(folder, img))
            print(f'→ trimmed to {target}')

        elif current < target:
            needed = target - current
            extras = random.choices(imgs, k=needed)
            for i, img in enumerate(extras):
                shutil.copy(
                    os.path.join(folder, img),
                    os.path.join(folder, f'dup_{i}_{img}')
                )
            print(f'→ duplicated to {target}')

        else:
            print('→ already balanced')

print('\nDone!') 
