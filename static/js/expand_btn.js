function toggleNews() {
    const moreNews = document.querySelectorAll('.more-news');
    const btn = document.getElementById('more-news-btn');
    
    moreNews.forEach(news => {
        if (news.style.display === "none") {
            news.style.display = "list-item";
            btn.textContent = "收起 | Collapse";
        } else {
            news.style.display = "none";
            btn.textContent = "查看更多 | See more";
        }
    });
}