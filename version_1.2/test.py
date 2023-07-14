import os
import pkg_resources


def calc_container(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


dists = [d for d in pkg_resources.working_set]

total_size_all_packages = 0  # Variable to store the total size

for dist in dists:
    try:
        path = os.path.join(dist.location, dist.project_name)
        size = calc_container(path)
        if size / 1000 > 1.0:
            total_size_all_packages += size  # Accumulate the size
            print(f"{dist}: {size / 1000} KB")
            print("-" * 40)
    except OSError:
        print('{} no longer exists'.format(dist.project_name))

# Print the total size of all packages
print(f"Total size of all packages: {total_size_all_packages / 1000} KB")
