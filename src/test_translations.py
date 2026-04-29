from src.augment import backtranslate_en, backtranslate_ko

sample_text = "Kumain ako ng tinapay"

print("Original:", sample_text)

# English backtranslation
en = backtranslate_en(sample_text)
print("\nEN backtranslation:")
print(en)

# Korean backtranslation
ko = backtranslate_ko(sample_text)
print("\nKO backtranslation:")
print(ko)

# Round-trip checks (optional but useful)
en_roundtrip = backtranslate_en(sample_text)
print("\nEN round-trip:")
print(en_roundtrip)

ko_roundtrip = backtranslate_ko(sample_text)
print("\nKO round-trip:")
print(ko_roundtrip)
