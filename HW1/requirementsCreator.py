import re

#print working directory
import os
print(os.getcwd())

# Read the full list of installed packages and their versions
with open('all_packages.txt', 'r') as f:
    all_packages = f.readlines()
all_packages = [pkg.strip() for pkg in all_packages]

# Create a dictionary to easily look up versions
version_dict = {}
for pkg in all_packages:
    match = re.match(r"([^=@<>~]+)", pkg)
    if match:
        package_name = match.group(1).lower().replace("_", "-")
        version_dict[package_name] = pkg

# Read the clean list of required packages
with open('clean_reqs.txt', 'r') as f:
    clean_reqs = f.readlines()
clean_reqs = [req.strip().split('==')[0] for req in clean_reqs]

# Create the final, accurate requirements file
with open('requirements.txt', 'w') as f:
    print("--- Final requirements.txt created with exact versions ---")
    for req in clean_reqs:
        req_key = req.lower().replace("_", "-")
        if req_key in version_dict:
            final_entry = version_dict[req_key]
            f.write(final_entry + '\n')
            print(final_entry)
        else:
            # Fallback for packages pipreqs found but pip freeze didn't list in a standard way
            f.write(req + '\n')
            print(f"{req} (version not found, added without version)")