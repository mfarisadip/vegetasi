#!/bin/bash

# Script untuk deployment aplikasi Streamlit
# Klasifikasi Vegetasi dari Citra Sentinel-2

echo "🚀 Memulai proses deployment..."

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fungsi untuk menampilkan pesan berwarna
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cek apakah Python terinstall
if ! command -v python3 &> /dev/null; then
    print_error "Python3 tidak ditemukan. Silakan install Python3 terlebih dahulu."
    exit 1
fi

# Cek apakah pip terinstall
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 tidak ditemukan. Silakan install pip3 terlebih dahulu."
    exit 1
fi

# Fungsi untuk setup environment
setup_environment() {
    print_status "Setting up virtual environment..."
    
    # Buat virtual environment jika belum ada
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment dibuat."
    else
        print_warning "Virtual environment sudah ada."
    fi
    
    # Aktivasi virtual environment
    source venv/bin/activate
    print_success "Virtual environment diaktivasi."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    print_success "Dependencies berhasil diinstall."
}

# Fungsi untuk menjalankan aplikasi lokal
run_local() {
    print_status "Menjalankan aplikasi Streamlit secara lokal..."
    source venv/bin/activate
    streamlit run vegetasi.py
}

# Fungsi untuk validasi file
validate_files() {
    print_status "Validating required files..."
    
    required_files=("vegetasi.py" "requirements.txt" "README.md")
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "File $file tidak ditemukan!"
            exit 1
        else
            print_success "✓ $file"
        fi
    done
}

# Fungsi untuk cek syntax Python
check_syntax() {
    print_status "Checking Python syntax..."
    python3 -m py_compile vegetasi.py
    if [ $? -eq 0 ]; then
        print_success "Syntax Python valid."
    else
        print_error "Syntax error ditemukan di vegetasi.py"
        exit 1
    fi
}

# Fungsi untuk deployment ke Git
deploy_to_git() {
    print_status "Preparing for Git deployment..."
    
    # Cek apakah git sudah diinisialisasi
    if [ ! -d ".git" ]; then
        print_warning "Git repository belum diinisialisasi."
        read -p "Inisialisasi Git repository? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git init
            print_success "Git repository diinisialisasi."
        else
            print_warning "Deployment ke Git dibatalkan."
            return
        fi
    fi
    
    # Add dan commit files
    git add .
    git status
    
    echo
    read -p "Commit message: " commit_message
    
    if [ -z "$commit_message" ]; then
        commit_message="Update aplikasi klasifikasi vegetasi"
    fi
    
    git commit -m "$commit_message"
    
    print_success "Files berhasil di-commit."
    print_warning "Jangan lupa push ke GitHub untuk deployment ke Streamlit Cloud!"
    print_status "Gunakan: git push origin main"
}

# Fungsi untuk menampilkan panduan deployment
show_deployment_guide() {
    echo
    print_status "=== PANDUAN DEPLOYMENT KE STREAMLIT CLOUD ==="
    echo
    echo "1. Upload repository ke GitHub:"
    echo "   - Buat repository baru di GitHub"
    echo "   - Push semua file ke repository"
    echo
    echo "2. Deploy ke Streamlit Cloud:"
    echo "   - Kunjungi: https://share.streamlit.io"
    echo "   - Login dengan akun GitHub"
    echo "   - Klik 'New app'"
    echo "   - Pilih repository dan file: vegetasi.py"
    echo "   - Klik 'Deploy!'"
    echo
    echo "3. URL aplikasi akan tersedia dalam 2-5 menit"
    echo
    print_success "Aplikasi siap untuk deployment!"
}

# Menu utama
show_menu() {
    echo
    echo "=== DEPLOYMENT SCRIPT - KLASIFIKASI VEGETASI ==="
    echo "1. Setup Environment"
    echo "2. Run Local"
    echo "3. Validate Files"
    echo "4. Check Syntax"
    echo "5. Prepare Git Deployment"
    echo "6. Show Deployment Guide"
    echo "7. Full Setup (1+3+4+5)"
    echo "0. Exit"
    echo
}

# Main script
while true; do
    show_menu
    read -p "Pilih opsi (0-7): " choice
    
    case $choice in
        1)
            setup_environment
            ;;
        2)
            run_local
            ;;
        3)
            validate_files
            ;;
        4)
            check_syntax
            ;;
        5)
            deploy_to_git
            ;;
        6)
            show_deployment_guide
            ;;
        7)
            validate_files
            check_syntax
            setup_environment
            deploy_to_git
            show_deployment_guide
            ;;
        0)
            print_success "Deployment script selesai. Terima kasih!"
            exit 0
            ;;
        *)
            print_error "Opsi tidak valid. Silakan pilih 0-7."
            ;;
    esac
    
    echo
    read -p "Tekan Enter untuk melanjutkan..."
done