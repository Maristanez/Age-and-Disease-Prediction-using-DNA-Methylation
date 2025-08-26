setlocal enabledelayedexpansion

REM ======================
REM Disease Datasets
REM ======================
set GEO_DISEASE=GSE144858 GSE153712

echo Creating directories for Disease...
for %%D in ("Disease\GEOdata" "Disease\IDMaps" "Disease\datasets" "Disease\Raw_IDAT" "Disease\IDAT_Data") do (
    if not exist "%%~D" mkdir "%%~D"
)

echo Downloading GEO data for Disease...
python downloadGEO.py Disease\GEOdata %GEO_DISEASE%

echo Creating ID maps...
python IDmaps.py Disease\IDMaps Disease\GEOdata %GEO_DISEASE%


echo Creating datasets...
python createGSEdatasets.py Disease\datasets Disease\GEOdata GSE144858

echo Downloading raw IDAT tarball for GSE153712...
curl -L -o Disease\Raw_IDAT\GSE153712_RAW.tar ^
    https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153712/suppl/GSE153712_RAW.tar

echo Processing IDAT files...
Rscript ProcessIDATFiles.R Disease\datasets Disease\IDAT_Data Disease\Raw_IDAT GSE153712

echo Building disease ID maps and merging datasets...
python diseaseIDmap.py Disease\disease_idmap.csv Disease\IDMaps %GEO_DISEASE%
python mergeCSVFiles.py Disease\disease_methylation_data Disease\datasets %GEO_DISEASE%
python getSiteList.py Disease\disease_CpG_sites.txt Disease\datasets %GEO_DISEASE%

pause


:skip1
REM ======================
REM Age Datasets
REM ======================
set GEO_AGE=GSE51032 GSE73103 GSE64495 GSE42861 GSE40279 GSE69270 GSE41169 GSE30870 GSE144858


echo Creating directories for Age...
for %%D in ("Age\GEOdata" "Age\IDMaps" "Age\datasets") do (
    if not exist "%%~D" mkdir "%%~D"
)

echo Downloading GEO data for Age...
python downloadGEO.py Age\GEOdata %GEO_AGE%

echo Creating ID maps...
python IDmaps.py Age\IDMaps Age\GEOdata %GEO_AGE%

echo Creating datasets...
python createGSEdatasets.py Age\datasets Age\GEOdata %GEO_AGE%

echo Building age ID maps and merging datasets...
python ageIDmap.py Age\age_idmap.csv Age\IDMaps %GEO_AGE%
python mergeCSVFiles.py Age\age_methylation_data Age\datasets %GEO_AGE%
python getSiteList.py Age\age_CpG_sites.txt Age\datasets %GEO_AGE%

echo.
echo ======================
echo ALL TASKS COMPLETED
echo ======================
pause
