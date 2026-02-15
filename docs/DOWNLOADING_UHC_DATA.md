# Downloading UHC Transparency in Coverage Data

## Automated Download

I've created a script to automatically download UHC MRF files:

```bash
# Install required packages
pip install requests beautifulsoup4 lxml

# Download MRF files (limits to 10 files by default)
python src/download_uhc_tic_data.py

# Download all MRF files
python src/download_uhc_tic_data.py --download-all

# Download specific number of files
python src/download_uhc_tic_data.py --max-files 5
```

The script will:
1. Search the UHC transparency website for MRF file links
2. Download MRF index files
3. Extract individual file URLs from index
4. Download all MRF files to `data/raw/`

## Manual Download (If Automated Fails)

If the automated script doesn't work (website structure may vary), you can download manually:

### Step 1: Visit UHC Transparency Site
1. Go to: https://transparency-in-coverage.uhc.com/
2. Navigate to "Machine Readable Files" section
3. Look for download links or API endpoints

### Step 2: Download MRF Index File
- Usually named `index.json` or similar
- Contains list of all MRF files
- May be in a subdirectory like `/mrf/` or `/files/`

### Step 3: Download Individual MRF Files
- Files are typically JSON format
- Can be very large (GBs)
- May need to download in chunks

### Step 4: Place Files in Project
```bash
# Create raw data directory
mkdir -p data/raw

# Move downloaded files
mv ~/Downloads/uhc_mrf_*.json data/raw/
```

## Using Downloaded Files

Once you have the MRF files:

```bash
# Load and integrate with NPPES data
python src/load_uhc_tic_data.py data/raw/uhc_mrf_index.json

# Or use a specific MRF file
python src/load_uhc_tic_data.py data/raw/in-network-rates.json
```

## Troubleshooting

### Script Can't Find Files
- Website structure may have changed
- May require authentication
- Try manual download instead

### Files Are Too Large
- MRF files can be several GBs
- Download may timeout
- Consider downloading in smaller chunks
- Use `wget` or `curl` for resumable downloads

### Rate Limiting
- Script includes delays between downloads
- If you get blocked, increase delay time
- Or download manually

## Alternative: Use Payerset

If UHC's site is difficult to access, consider using [Payerset](https://docs.payerset.com/) which provides:
- Pre-processed MRF data
- Multiple payer data sources
- Easier API access
- Data normalization

## File Structure

UHC MRF files typically contain:
- `reporting_structure`: Array of reporting entities
- `in_network_files`: List of in-network rate files
- `allowed_amount_files`: List of allowed amount files
- Each file entry has:
  - `location`: URL to the file
  - `description`: File description
  - `file_size_bytes`: File size
