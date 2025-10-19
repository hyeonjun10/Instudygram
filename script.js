// script.js
document.addEventListener('DOMContentLoaded', function () {
    const CSV_URL = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRZRfXqyANlWCyPV65IUS2IRwaczGQVBloK9P7B0nmNNW8u7eUn4r10fzj870hUPbZb-Qgm1LayxBli/pub?gid=1560776562&single=true&output=csv';

    // DOM Elements
    const mainContainer = document.getElementById('main-container');
    const mainContent = document.querySelector('.main-content');
    const paginationElement = document.getElementById('pagination');
    const filterContainer = document.querySelector('.filter-container');
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const sortAscBtn = document.getElementById('sort-asc-btn');
    const sortDescBtn = document.getElementById('sort-desc-btn');
    
    // Modal Elements
    const modalOverlay = document.getElementById('profile-modal-overlay');
    // --- 이 부분의 ID를 수정해서 오류를 해결했어요! ---
    const modalContainer = document.getElementById('profile-modal-container'); 
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalProfileCardPlaceholder = modalContainer.querySelector('.profile-card-placeholder');

    let allAcademyData = [];
    let currentPage = 1;
    const rowsPerPage = 10;
    let currentRegion = '전체';
    let searchQuery = '';

    // Kakao Map & Data Fetching
    let kakaoReadyPromise = null;
    function kakaoReady() { if (kakaoReadyPromise) return kakaoReadyPromise; kakaoReadyPromise = new Promise((resolve, reject) => { try { if (typeof window.kakao === 'undefined' || !window.kakao.maps) { return reject(new Error('카카오 지도 스크립트를 불러오지 못했습니다.')); } if (kakao.maps.load) kakao.maps.load(() => resolve()); else resolve(); } catch (e) { reject(e); } }); return kakaoReadyPromise; }
    const coordCache = new Map();
    let geocoder = null;
    async function geocode(address) { if (!address) throw new Error('주소가 없습니다.'); if (coordCache.has(address)) return coordCache.get(address); await kakaoReady(); if (!geocoder) geocoder = new kakao.maps.services.Geocoder(); return new Promise((resolve, reject) => { geocoder.addressSearch(address, (result, status) => { if (status === kakao.maps.services.Status.OK && result && result[0]) { const coords = { lat: parseFloat(result[0].y), lng: parseFloat(result[0].x) }; coordCache.set(address, coords); resolve(coords); } else { reject(new Error(`지오코딩 실패: ${status}`)); } }); }); }
    function fetchSheetCsv(url) { return new Promise((resolve, reject) => { Papa.parse(url, { download: true, header: true, skipEmptyLines: true, complete: (res) => { if (res && res.data && res.data.length) resolve(res.data); else reject(new Error('CSV 데이터가 비어있음')); }, error: (err) => reject(err) }); }); }
    function normalizeRows(raw) {
        const pick = (o, keys) => { for (const k of keys) { if (k in o && String(o[k]).trim()) return String(o[k]).trim(); } return ''; };
        return raw.map((o, i) => ({
            name: pick(o, ['학원명']), address: pick(o, ['도로명주소']), id: `academy_${i + 1}`,
            region: pick(o, ['행정구역명']), category: pick(o, ['교습계열명']), type: pick(o, ['도로명주소구분', '구분']),
            subject: pick(o, ['과목명']), grade: pick(o, ['성적구분']), scale: pick(o, ['학원규모']), weekend: pick(o, ['주말반']),
        })).filter(r => r.name && r.region);
    }
    
    // --- Modal Functions ---
    function showProfileModal(academy) { 
        const initial = academy.name ? academy.name.charAt(0).toUpperCase() : '?';
        modalProfileCardPlaceholder.innerHTML = `
            <div class="profile-card">
                <div class="avatar-container">
                    <img class="avatar" src="https://placehold.co/150x150/EFEFEF/3A3A3A?text=${encodeURIComponent(initial)}" alt="${escapeHtml(academy.name)} 로고">
                </div>
                <div class="info-container">
                    <div class="info-header">
                        <h1>${escapeHtml(academy.name)}</h1>
                        <img class="verified-badge" src="https://img.icons8.com/color/48/verified-badge.png" alt="인증 배지">
                    </div>
                    <div class="info-body">
                        <p><strong>주소</strong>: ${escapeHtml(academy.address)}</p>
                        <p><strong>평점</strong>: 4.5/5.0 (<a href="#">댓글 확인하기</a>)</p>
                        <p><strong>교습과목</strong>: ${escapeHtml(academy.subject)}</p>
                        <p><strong>대상</strong>: ${escapeHtml(academy.grade)}</p>
                        <p><strong>학원규모</strong>: ${escapeHtml(academy.scale)}</p>
                    </div>
                    <div class="info-footer">
                        <button class="consult-button">상담 신청</button>
                    </div>
                </div>
            </div>`;
        mainContainer.classList.add('modal-open-blur');
        modalOverlay.classList.remove('hidden');
     }
    function hideProfileModal() { 
        mainContainer.classList.remove('modal-open-blur');
        modalOverlay.classList.add('hidden');
     }
    modalOverlay.addEventListener('click', (e) => { if (e.target === modalOverlay) { hideProfileModal(); }});
    modalCloseBtn.addEventListener('click', hideProfileModal);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modalOverlay.classList.contains('hidden')) { hideProfileModal(); }});

    // --- Sort Button Event Listeners ---
    sortAscBtn.addEventListener('click', () => { allAcademyData.sort((a, b) => a.name.localeCompare(b.name, 'ko')); currentPage = 1; updateView(); });
    sortDescBtn.addEventListener('click', () => { allAcademyData.sort((a, b) => b.name.localeCompare(a.name, 'ko')); currentPage = 1; updateView(); });

    // --- Post Creation ---
    function createPostElement(academy) {
        const element = document.createElement('div');
        element.className = 'post-container';
        const initial = academy.name ? academy.name.charAt(0).toUpperCase() : '?';
        element.innerHTML = `
            <header class="post-header">
                <div class="post-author">
                    <img class="author-avatar" src="https://placehold.co/32x32/ffbbdd/4A4A4A?text=${encodeURIComponent(initial)}" alt="profile picture">
                    <span class="author-name">${escapeHtml(academy.name)}</span>
                </div>
            </header>
            <div class="address-display"><p>${escapeHtml(academy.address || '주소 정보 없음')}</p></div>
            <div class="map"></div>
            <div class="post-actions">
                 <div class="action-buttons-left">
                    <svg class="action-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M4.318 6.318a4 4 0 000 6.364L12 20.364l7.682-7.682a4 4 0 00-6.364-6.364L12 7.636l-1.318-1.318a4 4 0 00-6.364 0z"></path></svg>
                    <img class="action-icon" src="https://drive.google.com/thumbnail?id=1yn5SCtooNy_Vpr0H_C2mtjSdM7vGhrFo" alt="댓글">
                </div>
            </div>`;
        
        element.querySelector('.author-name').addEventListener('click', () => showProfileModal(academy));
        return element;
    }
    
    // --- Other Functions ---
    function setupFilters() { if (!filterContainer) return; const regions = ['전체', ...new Set(allAcademyData.map(d => d.region))]; filterContainer.innerHTML = ''; regions.forEach(region => { const button = document.createElement('button'); button.className = 'filter-btn'; button.innerText = region; if (region === currentRegion) button.classList.add('active'); button.addEventListener('click', () => { currentRegion = region; currentPage = 1; updateView(); }); filterContainer.appendChild(button); }); }
    searchButton.addEventListener('click', () => { searchQuery = searchInput.value.trim(); currentPage = 1; updateView(); });
    searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') searchButton.click(); });
    function updateView() { let filteredData = currentRegion === '전체' ? allAcademyData : allAcademyData.filter(d => d.region === currentRegion); if (searchQuery) { filteredData = filteredData.filter(d => d.name.toLowerCase().includes(searchQuery.toLowerCase())); } setupActiveFilters(); displayPosts(currentPage, filteredData); setupPagination(filteredData); }
    function setupActiveFilters() { document.querySelectorAll('.filter-btn').forEach(btn => { if (btn.innerText === currentRegion) btn.classList.add('active'); else btn.classList.remove('active'); }); }
    
    function displayPosts(page, data) {
        mainContent.innerHTML = "";
        page--;
        const start = rowsPerPage * page;
        const end = start + rowsPerPage;
        const paginatedItems = data.slice(start, end);

        if (paginatedItems.length === 0) {
            mainContent.innerHTML = `<p style="text-align:center;">결과에 해당하는 학원이 없습니다.</p>`;
        } else {
            paginatedItems.forEach(academy => {
                const postElement = createPostElement(academy);
                mainContent.appendChild(postElement);
                const mapContainer = postElement.querySelector('.map');
                placeMap(mapContainer, academy.address);
            });
        }
        document.querySelector('.main-content-wrapper').scrollTop = 0;
    }
    function setupPagination(data) { paginationElement.innerHTML = ""; const pageCount = Math.ceil(data.length / rowsPerPage); if (pageCount <= 1) return; const createPageButton = (page, text = page) => { const button = document.createElement('button'); button.classList.add('page-btn'); button.innerText = text; if (currentPage === page) button.classList.add('active'); button.addEventListener('click', () => { currentPage = page; updateView(); }); return button; }; const prevBtn = createPageButton(currentPage - 1, '‹'); if (currentPage === 1) prevBtn.disabled = true; paginationElement.appendChild(prevBtn); const pagesToShow = 5; let startPage = Math.max(1, currentPage - Math.floor(pagesToShow / 2)); let endPage = Math.min(pageCount, startPage + pagesToShow - 1); if (endPage - startPage + 1 < pagesToShow) startPage = Math.max(1, endPage - pagesToShow + 1); if (startPage > 1) { paginationElement.appendChild(createPageButton(1)); if (startPage > 2) paginationElement.appendChild(Object.assign(document.createElement('span'), { className: 'page-ellipsis', innerText: '...' })); } for (let i = startPage; i <= endPage; i++) paginationElement.appendChild(createPageButton(i)); if (endPage < pageCount) { if (endPage < pageCount - 1) paginationElement.appendChild(Object.assign(document.createElement('span'), { className: 'page-ellipsis', innerText: '...' })); paginationElement.appendChild(createPageButton(pageCount)); } const nextBtn = createPageButton(currentPage + 1, '›'); if (currentPage === pageCount) nextBtn.disabled = true; paginationElement.appendChild(nextBtn); }
    
    // --- 지도 생성 함수 (오류 메시지 강화) ---
    async function placeMap(container, address) {
        if (!container) return;
        container.innerHTML = `<div style="padding:18px;text-align:center;">지도 로딩중...</div>`;

        try {
            await kakaoReady();
            const coords = await geocode(address);
            container.innerHTML = '';
            const map = new kakao.maps.Map(container, { center: new kakao.maps.LatLng(coords.lat, coords.lng), level: 3 });
            new kakao.maps.Marker({ map, position: new kakao.maps.LatLng(coords.lat, coords.lng) });
        } catch (error) {
            console.error(`"${address}" 주소에 대한 지도 표시 오류:`, error);
            
            let userErrorMessage = '지도를 표시할 수 없습니다.';
            if (error.message.includes('ZERO_RESULT')) {
                userErrorMessage = '주소를 찾을 수 없어 지도를 표시할 수 없습니다.';
            } else if (error.message.includes('LIMIT')) {
                userErrorMessage = '지도 API 사용량을 초과했습니다. (카카오 개발자 사이트에서 확인 필요)';
            } else if (error.message.includes('지오코딩 실패')) {
                userErrorMessage = '주소 변환 중 오류가 발생했습니다.';
            }
            
            container.innerHTML = `<div style="padding:20px;text-align:center;">${userErrorMessage}</div>`;
        }
    }

    function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    (async function init() { try { mainContent.innerHTML = `<p style="text-align:center;">데이터를 불러오는 중...</p>`; const raw = await fetchSheetCsv(CSV_URL); allAcademyData = normalizeRows(raw); setupFilters(); updateView(); } catch (err) { console.error(err); mainContent.innerHTML = `<p style="text-align:center;color:red;">데이터를 불러오지 못했습니다.</p>`; } })();
});


